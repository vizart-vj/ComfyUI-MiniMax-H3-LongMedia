"""
Tests for refmod_backend.py — RefMod file-format/backend core for LongMedia.

Strict TDD: covers v4 standalone, v5 bundle, native-block conversion,
FULL/COMPRESSED modes, atomic save/load, metadata normalization, and
legacy tensor-key compat.

Attribution: test structure informed by Luisacaotica/ComfyUI-MiniMaxH3Mod
(MIT License).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

import refmod_backend
from refmod_backend import (
    META_KEY,
    RefMod,
    aspect_grid,
    compress_refmod,
    convert_to_native_block,
    encode_full,
    normalize_metadata,
    normalize_mode,
    pool_latent,
    read_refmod_meta,
    refine_latent,
    save_refmod,
    load_refmod,
    save_bundle,
    load_bundle,
)

# Real MiniMaxH3Mod artifact shipped with the reference implementation
_REFERENCE_EXAMPLE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "references", "ComfyUI-MiniMaxH3Mod", "mods",
    "vanellope_example.safetensors"))


# ── helpers ──────────────────────────────────────────────────────────────

def _make_image_latent(batch=1, ch=24, t=1, h=8, w=8):
    return torch.randn(batch, ch, t, h, w)

def _make_video_latent(batch=1, ch=24, t=4, h=8, w=8):
    return torch.randn(batch, ch, t, h, w)

def _make_audio_latent(batch=1, ch=32, stereo=2, t=16):
    return torch.randn(batch, ch, stereo, t)

def _make_refmod(name="test_ref", kind="image", **kwargs):
    defaults = dict(
        name=name,
        kind=kind,
        latent=_make_image_latent() if kind == "image" else
               _make_video_latent() if kind == "video" else
               _make_audio_latent(),
    )
    defaults.update(kwargs)
    return RefMod(**defaults)


def _write_raw(path_no_ext, metadata, tensors):
    """Write a safetensors file with an arbitrary raw header (compat fixtures)."""
    from safetensors.torch import save_file as sf
    dest = path_no_ext + ".safetensors"
    sf({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, dest,
       metadata={META_KEY: json.dumps(metadata)})
    return dest


# ═══════════════════════════════════════════════════════════════════════════
# A. RefMod creation & shape validation
# ═══════════════════════════════════════════════════════════════════════════

class TestRefModCreation(unittest.TestCase):
    """A1–A6: basic construction, shape invariants, and rejection of bad inputs."""

    def test_A1_image_latent_shape(self):
        """Image RefMod stores latent as [1,24,1,H,W]."""
        latent = _make_image_latent(t=1, h=12, w=10)
        r = RefMod(name="img", kind="image", latent=latent)
        self.assertEqual(r.kind, "image")
        self.assertEqual(r.latent_t, 1)
        self.assertEqual(r.latent_h, 12)
        self.assertEqual(r.latent_w, 10)
        self.assertEqual(r.latent.shape, (1, 24, 1, 12, 10))

    def test_A2_video_latent_shape(self):
        """Video RefMod stores latent as [1,24,T,H,W]."""
        latent = _make_video_latent(t=6, h=16, w=8)
        r = RefMod(name="vid", kind="video", latent=latent)
        self.assertEqual(r.kind, "video")
        self.assertEqual(r.latent_t, 6)
        self.assertEqual(r.latent_h, 16)
        self.assertEqual(r.latent_w, 8)

    def test_A3_audio_latent_shape(self):
        """Audio RefMod stores latent as [1,32,2,T]."""
        latent = _make_audio_latent(stereo=2, t=32)
        r = RefMod(name="aud", kind="audio", latent=latent)
        self.assertEqual(r.kind, "audio")
        self.assertEqual(r.latent_t, 32)
        self.assertEqual(r.latent_h, 0)
        self.assertEqual(r.latent_w, 0)

    def test_A4_image_forces_t1(self):
        """Image kind forces latent_t=1 regardless of input T."""
        latent = _make_image_latent(t=3)  # wrong T for image
        r = RefMod(name="fix", kind="image", latent=latent)
        self.assertEqual(r.latent_t, 1)

    def test_A5_rejects_invalid_kind(self):
        """Reject unknown kind values."""
        with self.assertRaises(ValueError):
            RefMod(name="bad", kind="audio_stream", latent=_make_image_latent())

    def test_A6_rejects_bad_audio_shape(self):
        """Audio latent must be [1,32,2,T]."""
        with self.assertRaises(ValueError):
            RefMod(name="bad", kind="audio", latent=torch.randn(1, 24, 4, 4, 4))

    def test_A7_rejects_wrong_visual_channels(self):
        """Visual latent must have exactly 24 channels."""
        with self.assertRaises(ValueError):
            RefMod(name="bad", kind="image", latent=torch.randn(1, 16, 1, 8, 8))

    def test_A8_rejects_wrong_visual_batch(self):
        """Visual latent must have batch 1."""
        with self.assertRaises(ValueError):
            RefMod(name="bad", kind="video", latent=torch.randn(2, 24, 4, 8, 8))

    def test_A9_rejects_visual_4d(self):
        """Visual latent must be 5-D [1,24,T,H,W]."""
        with self.assertRaises(ValueError):
            RefMod(name="bad", kind="image", latent=torch.randn(1, 24, 8, 8))


# ═══════════════════════════════════════════════════════════════════════════
# B. Metadata normalization
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeMetadata(unittest.TestCase):
    """B1–B4: safe metadata normalization from raw dicts."""

    def test_B1_normalizes_minimal(self):
        """A minimal metadata dict is filled with defaults."""
        meta = normalize_metadata({"name": "x", "kind": "image"})
        self.assertEqual(meta["_format_version"], 4)
        self.assertEqual(meta["kind"], "image")
        self.assertEqual(meta["mode"], "FULL")
        self.assertIn("latent_t", meta)

    def test_B2_unknown_keys_ignored(self):
        """Extra/unknown keys do not crash normalization."""
        meta = normalize_metadata({"name": "x", "kind": "video",
                                    "latent_t": 8, "bogus": 42})
        self.assertEqual(meta["latent_t"], 8)
        self.assertNotIn("bogus", meta)

    def test_B3_bundle_detected(self):
        """format_version=5 + kind=bundle is flagged."""
        meta = normalize_metadata({"_format_version": 5, "kind": "bundle",
                                    "name": "b", "members": []})
        self.assertTrue(meta.get("is_bundle"))

    def test_B4_default_concept_type(self):
        """concept_type defaults to 'generic' when absent."""
        meta = normalize_metadata({"name": "x", "kind": "image"})
        self.assertEqual(meta["concept_type"], "generic")

    def test_B5_legacy_mode_aliases(self):
        """MiniMaxH3Mod legacy modes map onto FULL / COMPRESSED."""
        self.assertEqual(normalize_mode("encode"), "FULL")
        self.assertEqual(normalize_mode("full"), "FULL")
        self.assertEqual(normalize_mode("Full Reference"), "FULL")
        self.assertEqual(normalize_mode("training"), "COMPRESSED")
        self.assertEqual(normalize_mode("pooled"), "COMPRESSED")
        self.assertEqual(normalize_mode("Compressed Reference"), "COMPRESSED")
        self.assertEqual(normalize_mode("FULL"), "FULL")
        self.assertEqual(normalize_mode("COMPRESSED"), "COMPRESSED")

    def test_B6_normalize_metadata_maps_mode(self):
        """normalize_metadata normalizes stored legacy mode names."""
        meta = normalize_metadata({"name": "x", "kind": "video", "mode": "pooled"})
        self.assertEqual(meta["mode"], "COMPRESSED")
        meta = normalize_metadata({"name": "x", "kind": "image", "mode": "encode"})
        self.assertEqual(meta["mode"], "FULL")

    def test_B7_refmod_accepts_legacy_mode(self):
        """A RefMod built with a legacy mode string is normalized."""
        r = RefMod(name="x", kind="image", latent=_make_image_latent(),
                   mode="training")
        self.assertEqual(r.mode, "COMPRESSED")
        r2 = RefMod(name="y", kind="image", latent=_make_image_latent(),
                    mode="encode")
        self.assertEqual(r2.mode, "FULL")


# ═══════════════════════════════════════════════════════════════════════════
# C. pool_latent & aspect_grid
# ═══════════════════════════════════════════════════════════════════════════

class TestPoolLatent(unittest.TestCase):
    """C1–C4: average-pool to a small grid."""

    def test_C1_pool_reduces_spatial(self):
        """pool_latent reduces H,W to target dims."""
        z = _make_image_latent(h=32, w=16)
        pooled = pool_latent(z, latent_t=1, latent_h=4, latent_w=8)
        self.assertEqual(pooled.shape, (1, 24, 1, 4, 8))

    def test_C2_pool_noop_when_exact(self):
        """pool_latent returns same tensor if dims already match."""
        z = _make_image_latent(h=4, w=4)
        pooled = pool_latent(z, latent_t=1, latent_h=4, latent_w=4)
        self.assertTrue(torch.equal(z, pooled))

    def test_C3_pool_preserves_dtype(self):
        """Pooling in fp32 still returns original dtype."""
        z = _make_image_latent(h=8, w=8).half()
        pooled = pool_latent(z, latent_t=1, latent_h=4, latent_w=4)
        self.assertEqual(pooled.dtype, torch.float16)

    def test_C4_pool_rejects_odd_dims(self):
        """Odd pool dims are rejected (DiT patches 2x2)."""
        z = _make_image_latent(h=8, w=8)
        with self.assertRaises(ValueError):
            pool_latent(z, latent_t=1, latent_h=3, latent_w=4)

    def test_C5_pool_temporal(self):
        """Pooling also reduces temporal dim."""
        z = _make_video_latent(t=12, h=16, w=16)
        pooled = pool_latent(z, latent_t=4, latent_h=4, latent_w=4)
        self.assertEqual(pooled.shape, (1, 24, 4, 4, 4))

    def test_C9_pool_rejects_nonpositive_target(self):
        """A non-positive pool target is rejected, not silently accepted."""
        z = _make_image_latent(h=8, w=8)
        with self.assertRaises(ValueError):
            pool_latent(z, latent_t=1, latent_h=0, latent_w=4)

    def test_C10_pool_audio_temporal_only(self):
        """Audio [1,32,2,T] pools along T only (COMPRESSED audio path)."""
        z = _make_audio_latent(t=64)
        pooled = pool_latent(z, latent_t=8, latent_h=0, latent_w=0)
        self.assertEqual(tuple(pooled.shape), (1, 32, 2, 8))
        self.assertEqual(pooled.dtype, z.dtype)

    def test_C11_pool_audio_noop_when_exact(self):
        """Audio pooling returns the same tensor when T already matches."""
        z = _make_audio_latent(t=16)
        self.assertTrue(torch.equal(pool_latent(z, 16, 0, 0), z))


class TestAspectGrid(unittest.TestCase):
    """C6–C8: aspect-preserving grid."""

    def test_C6_square_stays_square(self):
        """Square source keeps dial value."""
        h, w = aspect_grid(16, 16, 1.0)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)

    def test_C7_portrait_preserves_ratio(self):
        """Tall source (aspect > 1, i.e. h/w tall) gets a tall grid."""
        h, w = aspect_grid(16, 16, 2.0)  # 2:1 portrait (h>w)
        self.assertGreater(h, w)

    def test_C8_landscape_preserves_ratio(self):
        """Wide source (aspect < 1, i.e. h/w wide) gets a wide grid."""
        h, w = aspect_grid(16, 16, 0.5)  # 1:2 landscape (w>h)
        self.assertGreater(w, h)


# ═══════════════════════════════════════════════════════════════════════════
# D. refine_latent (Adam/MSE refinement)
# ═══════════════════════════════════════════════════════════════════════════

class TestRefineLatent(unittest.TestCase):
    """D1–D3: model-free MSE refinement of compressed latent."""

    def test_D1_refinement_reduces_loss(self):
        """After refinement, MSE to target is lower than before."""
        target = _make_image_latent(h=16, w=16)
        small = pool_latent(target, 1, 4, 4)
        up_before = F.interpolate(small.float(), size=(1, 16, 16),
                                  mode="trilinear", align_corners=False)
        loss_before = F.mse_loss(up_before, target.float()).item()

        refined = refine_latent(small, target, steps=30, lr=0.02)
        up_after = F.interpolate(refined.float(), size=(1, 16, 16),
                                 mode="trilinear", align_corners=False)
        loss_after = F.mse_loss(up_after, target.float()).item()

        self.assertLess(loss_after, loss_before)

    def test_D2_refinement_preserves_shape(self):
        """Refined tensor has same shape as input small."""
        target = _make_image_latent(h=16, w=16)
        small = pool_latent(target, 1, 4, 4)
        refined = refine_latent(small, target, steps=5, lr=0.01)
        self.assertEqual(refined.shape, small.shape)

    def test_D3_refinement_zero_steps(self):
        """Zero steps returns input unchanged."""
        z = _make_image_latent(h=4, w=4)
        result = refine_latent(z, z, steps=0)
        self.assertTrue(torch.equal(result, z))

    def test_D4_refinement_audio_linear(self):
        """Audio refinement works on the 4-D [1,32,2,T] layout."""
        target = _make_audio_latent(t=64)
        small = pool_latent(target, latent_t=8, latent_h=0, latent_w=0)
        up_before = F.interpolate(
            small.reshape(1 * 32 * 2, 1, 8).float(), size=64,
            mode="linear", align_corners=False).reshape_as(target)
        refined = refine_latent(small, target, steps=30, lr=0.02)
        self.assertEqual(refined.shape, small.shape)
        up_after = F.interpolate(
            refined.reshape(1 * 32 * 2, 1, 8).float(), size=64,
            mode="linear", align_corners=False).reshape_as(target)
        loss_before = F.mse_loss(up_before, target.float()).item()
        loss_after = F.mse_loss(up_after, target.float()).item()
        self.assertLess(loss_after, loss_before)


# ═══════════════════════════════════════════════════════════════════════════
# E. encode_full (FULL mode — no compression)
# ═══════════════════════════════════════════════════════════════════════════

class TestEncodeFull(unittest.TestCase):
    """E1–E2: FULL mode returns the original latent wrapped as RefMod."""

    def test_E1_encode_full_returns_original(self):
        """encode_full preserves the exact latent tensor."""
        latent = _make_image_latent(h=16, w=16)
        r = encode_full(latent, name="full_img", kind="image")
        self.assertTrue(torch.equal(r.latent, latent))
        self.assertEqual(r.mode, "FULL")

    def test_E2_encode_full_video(self):
        """encode_full works for video kind."""
        latent = _make_video_latent(t=8, h=8, w=8)
        r = encode_full(latent, name="full_vid", kind="video")
        self.assertEqual(r.latent_t, 8)
        self.assertEqual(r.kind, "video")


# ═══════════════════════════════════════════════════════════════════════════
# F. compress_refmod (COMPRESSED mode — pool + optional refine)
# ═══════════════════════════════════════════════════════════════════════════

class TestCompressRefmod(unittest.TestCase):
    """F1–F4: COMPRESSED mode pools and optionally refines."""

    def test_F1_compress_pools(self):
        """compress_refmod pools latent to target grid."""
        latent = _make_image_latent(h=32, w=32)
        r = compress_refmod(latent, name="c", kind="image",
                            pool_h=4, pool_w=4, refine_steps=0)
        self.assertEqual(r.latent.shape, (1, 24, 1, 4, 4))
        self.assertEqual(r.mode, "COMPRESSED")

    def test_F2_compress_with_refinement(self):
        """compress_refmod with refinement reduces MSE vs naive pool."""
        target = _make_image_latent(h=16, w=16)
        r_pooled = compress_refmod(target, name="p", kind="image",
                                    pool_h=4, pool_w=4, refine_steps=0)
        r_refined = compress_refmod(target, name="r", kind="image",
                                     pool_h=4, pool_w=4, refine_steps=30)
        up_p = F.interpolate(r_pooled.latent.float(), size=(1, 16, 16),
                              mode="trilinear", align_corners=False)
        up_r = F.interpolate(r_refined.latent.float(), size=(1, 16, 16),
                              mode="trilinear", align_corners=False)
        loss_p = F.mse_loss(up_p, target.float()).item()
        loss_r = F.mse_loss(up_r, target.float()).item()
        self.assertLess(loss_r, loss_p)

    def test_F3_compress_video(self):
        """compress_refmod works for video latent."""
        latent = _make_video_latent(t=8, h=32, w=16)
        r = compress_refmod(latent, name="cv", kind="video",
                            pool_t=4, pool_h=4, pool_w=4, refine_steps=0)
        self.assertEqual(r.latent_t, 4)
        self.assertEqual(r.latent_h, 4)
        self.assertEqual(r.latent_w, 4)

    def test_F4_compress_preserves_metadata(self):
        """Name, kind, source flow through to the RefMod."""
        latent = _make_image_latent(h=16, w=16)
        r = compress_refmod(latent, name="meta_test", kind="image",
                            pool_h=4, pool_w=4, refine_steps=0,
                            source="image")
        self.assertEqual(r.name, "meta_test")
        self.assertEqual(r.source, "image")

    def test_F5_compress_audio_temporal(self):
        """COMPRESSED audio pools only along T and keeps [1,32,2,T]."""
        latent = _make_audio_latent(t=64)
        r = compress_refmod(latent, name="ca", kind="audio",
                            pool_t=8, refine_steps=0)
        self.assertEqual(tuple(r.latent.shape), (1, 32, 2, 8))
        self.assertEqual(r.mode, "COMPRESSED")
        self.assertEqual(r.latent_t, 8)
        self.assertEqual(r.pool, "8x1x1")

    def test_F6_compress_audio_with_refinement(self):
        """Audio refinement lowers reconstruction MSE vs naive pooling."""
        target = _make_audio_latent(t=48)
        r_pooled = compress_refmod(target, name="ap", kind="audio",
                                   pool_t=6, refine_steps=0)
        r_refined = compress_refmod(target, name="ar", kind="audio",
                                    pool_t=6, refine_steps=30)

        def up(mod):
            return F.interpolate(
                mod.latent.reshape(1 * 32 * 2, 1, 6).float(), size=48,
                mode="linear", align_corners=False).reshape_as(target)

        loss_p = F.mse_loss(up(r_pooled), target.float()).item()
        loss_r = F.mse_loss(up(r_refined), target.float()).item()
        self.assertLess(loss_r, loss_p)

    def test_F7_image_forces_pool_t1(self):
        """Image compression always pools T to 1 regardless of pool_t."""
        latent = _make_image_latent(t=5, h=16, w=16)
        r = compress_refmod(latent, name="i", kind="image",
                            pool_t=4, pool_h=4, pool_w=4, refine_steps=0)
        self.assertEqual(tuple(r.latent.shape), (1, 24, 1, 4, 4))
        self.assertEqual(r.latent_t, 1)
        self.assertEqual(r.pool, "1x4x4")


# ═══════════════════════════════════════════════════════════════════════════
# G. Atomic safetensors save/load — v4 standalone
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveLoadV4(unittest.TestCase):
    """G1–G7: v4 standalone atomic save/load round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "test_ref")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_G1_save_creates_safetensors(self):
        """save_refmod writes a .safetensors file."""
        r = _make_refmod()
        dest = save_refmod(r, self.path)
        self.assertTrue(os.path.isfile(dest))
        self.assertTrue(dest.endswith(".safetensors"))

    def test_G2_load_roundtrip_image(self):
        """save + load image RefMod preserves all fields."""
        latent = _make_image_latent(h=16, w=12)
        r = RefMod(name="roundtrip", kind="image", latent=latent,
                    description="test desc", concept_type="identity",
                    tags=["a", "b"])
        save_refmod(r, self.path)
        r2 = load_refmod(self.path)
        self.assertEqual(r2.name, "roundtrip")
        self.assertEqual(r2.kind, "image")
        self.assertTrue(torch.equal(r2.latent, latent))
        self.assertEqual(r2.description, "test desc")
        self.assertEqual(r2.concept_type, "identity")
        self.assertEqual(r2.tags, ["a", "b"])

    def test_G3_load_roundtrip_video(self):
        """save + load video RefMod preserves T, H, W."""
        latent = _make_video_latent(t=10, h=8, w=8)
        r = RefMod(name="vid", kind="video", latent=latent)
        save_refmod(r, self.path)
        r2 = load_refmod(self.path)
        self.assertEqual(r2.latent_t, 10)
        self.assertEqual(r2.latent_h, 8)
        self.assertEqual(r2.latent_w, 8)
        self.assertTrue(torch.equal(r2.latent, latent))

    def test_G4_load_roundtrip_audio(self):
        """save + load audio RefMod preserves [1,32,2,T]."""
        latent = _make_audio_latent(t=64)
        r = RefMod(name="aud", kind="audio", latent=latent)
        save_refmod(r, self.path)
        r2 = load_refmod(self.path)
        self.assertEqual(r2.latent.shape, (1, 32, 2, 64))
        self.assertEqual(r2.kind, "audio")
        self.assertTrue(torch.equal(r2.latent, latent))

    def test_G5_metadata_key_in_header(self):
        """refmod_meta key is present in safetensors header."""
        r = _make_refmod()
        dest = save_refmod(r, self.path)
        with safe_open(dest, framework="pt") as f:
            meta = f.metadata()
        self.assertIn(META_KEY, meta)
        parsed = json.loads(meta[META_KEY])
        self.assertEqual(parsed["kind"], "image")

    def test_G6_legacy_tensor_key_loads(self):
        """Loading a file with tensor key 'latent' (legacy) works."""
        r = _make_refmod()
        # Manually save with legacy key
        meta = r.metadata_dict()
        meta["_format_version"] = 4
        dest = self.path + ".safetensors"
        from safetensors.torch import save_file as sf
        sf({"latent": r.latent.contiguous()}, dest,
           metadata={META_KEY: json.dumps(meta)})
        r2 = load_refmod(self.path)
        self.assertTrue(torch.equal(r2.latent, r.latent))

    def test_G7_canonical_tensor_key_loads(self):
        """Loading a file with tensor key 'ref_0' (canonical) works."""
        r = _make_refmod()
        meta = r.metadata_dict()
        meta["_format_version"] = 4
        dest = self.path + ".safetensors"
        from safetensors.torch import save_file as sf
        sf({"ref_0": r.latent.contiguous()}, dest,
           metadata={META_KEY: json.dumps(meta)})
        r2 = load_refmod(self.path)
        self.assertTrue(torch.equal(r2.latent, r.latent))

    def test_G8_canonical_key_written(self):
        """save_refmod writes the canonical v4 tensor key 'ref_0'."""
        r = _make_refmod()
        dest = save_refmod(r, self.path)
        with safe_open(dest, framework="pt") as f:
            self.assertEqual(list(f.keys()), ["ref_0"])

    def test_G9_full_mode_roundtrip(self):
        """FULL mode survives save/load with mode and exact latent intact."""
        latent = _make_image_latent(h=16, w=16)
        r = encode_full(latent, name="full_img", kind="image")
        save_refmod(r, self.path)
        r2 = load_refmod(self.path)
        self.assertEqual(r2.mode, "FULL")
        self.assertTrue(torch.equal(r2.latent, latent))

    def test_G10_load_refmod_rejects_bundle_path(self):
        """load_refmod on a bundle path points the caller at load_bundle."""
        save_bundle([_make_refmod()], "b", self.path)
        with self.assertRaises(ValueError):
            load_refmod(self.path)


# ═══════════════════════════════════════════════════════════════════════════
# H. read_refmod_meta (header-only read)
# ═══════════════════════════════════════════════════════════════════════════

class TestReadRefmodMeta(unittest.TestCase):
    """H1–H3: reading metadata without loading tensors."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "meta_test")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_H1_read_meta_from_safetensors(self):
        """read_refmod_meta extracts metadata from safetensors header."""
        r = _make_refmod()
        save_refmod(r, self.path)
        meta = read_refmod_meta(self.path)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["kind"], "image")
        self.assertEqual(meta["_format_version"], 4)

    def test_H2_read_meta_missing_file(self):
        """read_refmod_meta returns None for nonexistent path."""
        meta = read_refmod_meta("/nonexistent/path/ref")
        self.assertIsNone(meta)

    def test_H3_read_meta_bundle(self):
        """read_refmod_meta detects v5 bundles."""
        # Save a bundle first
        refs = [
            _make_refmod(name="a", kind="image"),
            _make_refmod(name="b", kind="video", latent=_make_video_latent()),
        ]
        save_bundle(refs, "test_bundle", self.path)
        meta = read_refmod_meta(self.path)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["kind"], "bundle")
        self.assertEqual(meta["_format_version"], 5)


# ═══════════════════════════════════════════════════════════════════════════
# I. v5 bundle save/load
# ═══════════════════════════════════════════════════════════════════════════

class TestBundle(unittest.TestCase):
    """I1–I6: v5 bundle format with ref_0..ref_N tensor keys."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "bundle")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_I1_save_bundle_creates_file(self):
        """save_bundle writes a single .safetensors."""
        refs = [_make_refmod(name="a", kind="image"),
                _make_refmod(name="b", kind="image")]
        dest = save_bundle(refs, "mybundle", self.path)
        self.assertTrue(os.path.isfile(dest))

    def test_I2_bundle_tensor_keys(self):
        """Bundle tensors are keyed ref_0, ref_1, ..."""
        refs = [_make_refmod(name="a", kind="image"),
                _make_refmod(name="b", kind="audio",
                             latent=_make_audio_latent())]
        save_bundle(refs, "b", self.path)
        with safe_open(self.path + ".safetensors", framework="pt") as f:
            keys = [k for k in f.keys()]
        self.assertIn("ref_0", keys)
        self.assertIn("ref_1", keys)

    def test_I3_bundle_metadata(self):
        """Bundle header has kind=bundle, _format_version=5, members list."""
        refs = [_make_refmod(name="x")]
        save_bundle(refs, "b", self.path)
        with safe_open(self.path + ".safetensors", framework="pt") as f:
            meta = json.loads(f.metadata()[META_KEY])
        self.assertEqual(meta["kind"], "bundle")
        self.assertEqual(meta["_format_version"], 5)
        self.assertIsInstance(meta["members"], list)
        self.assertEqual(len(meta["members"]), 1)

    def test_I4_load_bundle_all(self):
        """load_bundle with selection='All' returns all members."""
        refs = [_make_refmod(name="a", kind="image"),
                _make_refmod(name="b", kind="video",
                             latent=_make_video_latent())]
        save_bundle(refs, "b", self.path)
        loaded = load_bundle(self.path, selection="All")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0][0].name, "a")
        self.assertEqual(loaded[1][0].name, "b")

    def test_I5_load_bundle_visual_only(self):
        """load_bundle selection='Visual' excludes audio."""
        refs = [_make_refmod(name="v", kind="image"),
                _make_refmod(name="a", kind="audio",
                             latent=_make_audio_latent())]
        save_bundle(refs, "b", self.path)
        loaded = load_bundle(self.path, selection="Visual")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0].kind, "image")

    def test_I6_load_bundle_audio_only(self):
        """load_bundle selection='Audio' includes only audio."""
        refs = [_make_refmod(name="v", kind="image"),
                _make_refmod(name="a", kind="audio",
                             latent=_make_audio_latent())]
        save_bundle(refs, "b", self.path)
        loaded = load_bundle(self.path, selection="Audio")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0].kind, "audio")

    def test_I7_save_rejects_empty_bundle(self):
        """An empty bundle is rejected at save time."""
        with self.assertRaises(ValueError):
            save_bundle([], "empty", self.path)

    def test_I8_save_rejects_oversized_bundle(self):
        """Bundles over 256 members are rejected (MiniMaxH3Mod limit)."""
        refs = [_make_refmod(name="r")] * 257
        with self.assertRaises(ValueError):
            save_bundle(refs, "huge", self.path)

    def test_I9_load_rejects_missing_members(self):
        """A bundle header without a members list is rejected."""
        metadata = {"_format_version": 5, "kind": "bundle", "name": "b"}
        _write_raw(self.path, metadata, {"ref_0": _make_refmod().latent})
        with self.assertRaises(ValueError):
            load_bundle(self.path)

    def test_I10_load_rejects_empty_members(self):
        """A bundle header with an empty members list is rejected."""
        metadata = {"_format_version": 5, "kind": "bundle", "name": "b",
                    "members": []}
        _write_raw(self.path, metadata, {"ref_0": _make_refmod().latent})
        with self.assertRaises(ValueError):
            load_bundle(self.path)

    def test_I11_load_rejects_bad_member_kind(self):
        """A bundle member with an unknown kind is rejected."""
        good = _make_refmod(name="ok")
        member = good.metadata_dict()
        member["kind"] = "model"
        metadata = {"_format_version": 5, "kind": "bundle", "name": "b",
                    "members": [member]}
        _write_raw(self.path, metadata, {"ref_0": good.latent})
        with self.assertRaises(ValueError):
            load_bundle(self.path)

    def test_I12_load_rejects_invalid_selection(self):
        """Unknown selection strings are rejected."""
        save_bundle([_make_refmod()], "b", self.path)
        with self.assertRaises(ValueError):
            load_bundle(self.path, selection="Video")

    def test_I13_load_rejects_out_of_range_strength(self):
        """Strengths outside [0, 1] or non-finite are rejected."""
        save_bundle([_make_refmod()], "b", self.path)
        with self.assertRaises(ValueError):
            load_bundle(self.path, visual_strength=1.5)
        with self.assertRaises(ValueError):
            load_bundle(self.path, audio_strength=-0.1)
        with self.assertRaises(ValueError):
            load_bundle(self.path, visual_strength=float("nan"))

    def test_I14_zero_strength_drops_only_that_kind(self):
        """visual_strength=0 drops visual members, keeps audio."""
        refs = [_make_refmod(name="v", kind="image"),
                _make_refmod(name="a", kind="audio",
                             latent=_make_audio_latent())]
        save_bundle(refs, "b", self.path)
        loaded = load_bundle(self.path, visual_strength=0.0)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0].kind, "audio")

    def test_I15_load_bundle_rejects_standalone_path(self):
        """load_bundle on a v4 standalone file is rejected."""
        save_refmod(_make_refmod(), self.path)
        with self.assertRaises(ValueError):
            load_bundle(self.path)


# ═══════════════════════════════════════════════════════════════════════════
# J. convert_to_native_block
# ═══════════════════════════════════════════════════════════════════════════

class TestConvertToNativeBlock(unittest.TestCase):
    """J1–J4: conversion to native minimax_refs block dicts."""

    def test_J1_image_block(self):
        """Image RefMod -> native image block."""
        latent = _make_image_latent(h=12, w=10)
        r = RefMod(name="img", kind="image", latent=latent)
        block = convert_to_native_block(r)
        self.assertEqual(block["kind"], "image")
        self.assertEqual(block["latent_h"], 12)
        self.assertEqual(block["latent_w"], 10)
        self.assertTrue(torch.equal(block["latent"], latent))

    def test_J2_video_block(self):
        """Video RefMod -> native video block with latent_t, ref_audio_t."""
        latent = _make_video_latent(t=6, h=8, w=8)
        r = RefMod(name="vid", kind="video", latent=latent)
        block = convert_to_native_block(r)
        self.assertEqual(block["kind"], "video")
        self.assertEqual(block["latent_t"], 6)
        self.assertEqual(block["ref_audio_t"], 0)
        self.assertIsNone(block["audio_latent"])

    def test_J3_audio_block(self):
        """Audio RefMod -> native audio block."""
        latent = _make_audio_latent(t=32)
        r = RefMod(name="aud", kind="audio", latent=latent)
        block = convert_to_native_block(r)
        self.assertEqual(block["kind"], "audio")
        self.assertEqual(block["ref_audio_t"], 32)
        self.assertTrue(torch.equal(block["audio_latent"], latent))

    def test_J4_strength_weakens_latent(self):
        """convert_to_native_block with strength < 1 applies blur."""
        latent = _make_image_latent(h=16, w=16)
        r = RefMod(name="weak", kind="image", latent=latent)
        block_strong = convert_to_native_block(r, strength=1.0)
        block_weak = convert_to_native_block(r, strength=0.5)
        # Different tensors — the weak one should be blurred
        self.assertFalse(torch.equal(block_strong["latent"],
                                      block_weak["latent"]))

    def test_J5_strength_zero_returns_none(self):
        """strength <= 0 drops the block."""
        r = _make_refmod()
        block = convert_to_native_block(r, strength=0.0)
        self.assertIsNone(block)

    def test_J6_exact_image_block_keys(self):
        """Image block carries exactly the native key set."""
        block = convert_to_native_block(_make_refmod(kind="image"))
        self.assertEqual(set(block), {"kind", "latent_h", "latent_w", "latent"})

    def test_J7_exact_video_block_keys(self):
        """Video block carries exactly the native key set."""
        block = convert_to_native_block(
            _make_refmod(kind="video", latent=_make_video_latent()))
        self.assertEqual(set(block), {"kind", "latent_h", "latent_w", "latent",
                                      "latent_t", "ref_audio_t", "audio_latent"})

    def test_J8_exact_audio_block_keys(self):
        """Audio block carries exactly the native key set."""
        block = convert_to_native_block(
            _make_refmod(kind="audio", latent=_make_audio_latent()))
        self.assertEqual(set(block), {"kind", "ref_audio_t", "audio_latent"})

    def test_J9_block_shapes_match_tensor(self):
        """Native block dims always mirror the stored tensor layout."""
        vid = _make_refmod(kind="video", latent=_make_video_latent(t=5, h=6, w=8))
        blk = convert_to_native_block(vid)
        self.assertEqual((blk["latent_t"], blk["latent_h"], blk["latent_w"]),
                         tuple(blk["latent"].shape[2:]))
        self.assertEqual(tuple(blk["latent"].shape[:2]), (1, 24))
        aud = _make_refmod(kind="audio", latent=_make_audio_latent(t=24))
        ablk = convert_to_native_block(aud)
        self.assertEqual(tuple(ablk["audio_latent"].shape), (1, 32, 2, 24))
        self.assertEqual(ablk["ref_audio_t"], 24)

    def test_J10_strength_one_is_identity(self):
        """strength=1.0 must not touch the latent at all."""
        r = _make_refmod(kind="video", latent=_make_video_latent())
        blk = convert_to_native_block(r, strength=1.0)
        self.assertTrue(torch.equal(blk["latent"], r.latent))


# ═══════════════════════════════════════════════════════════════════════════
# K. Atomicity & edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestAtomicity(unittest.TestCase):
    """K1–K2: atomic save (temp file + rename)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "atomic")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_K1_no_temp_files_after_save(self):
        """After save_refmod, no .tmp/.refmod-* files remain."""
        r = _make_refmod()
        save_refmod(r, self.path)
        remaining = [f for f in os.listdir(self.tmpdir)
                     if f.startswith(".refmod-") or f.endswith(".tmp")]
        self.assertEqual(remaining, [])

    def test_K2_overwrite_existing(self):
        """Saving to an existing path replaces it cleanly."""
        r1 = _make_refmod(name="v1")
        save_refmod(r1, self.path)
        r2 = _make_refmod(name="v2")
        save_refmod(r2, self.path)
        r3 = load_refmod(self.path)
        self.assertEqual(r3.name, "v2")

    def test_K3_failed_save_keeps_previous_file(self):
        """A failed write never clobbers the previously saved file."""
        save_refmod(_make_refmod(name="original"), self.path)
        original = refmod_backend.save_file

        def boom(*args, **kwargs):
            raise OSError("simulated disk failure")

        refmod_backend.save_file = boom
        try:
            with self.assertRaises(OSError):
                save_refmod(_make_refmod(name="replacement"), self.path)
        finally:
            refmod_backend.save_file = original
        self.assertEqual(load_refmod(self.path).name, "original")
        remaining = [f for f in os.listdir(self.tmpdir)
                     if f.startswith(".refmod-") or f.endswith(".tmp")]
        self.assertEqual(remaining, [])

    def test_K4_failed_bundle_save_cleans_temp(self):
        """A failed bundle write leaves no temp files behind."""
        save_bundle([_make_refmod()], "b", self.path)
        original = refmod_backend.save_file

        def boom(*args, **kwargs):
            raise OSError("simulated disk failure")

        refmod_backend.save_file = boom
        try:
            with self.assertRaises(OSError):
                save_bundle([_make_refmod()], "b2", self.path)
        finally:
            refmod_backend.save_file = original
        self.assertEqual(len(load_bundle(self.path)), 1)
        remaining = [f for f in os.listdir(self.tmpdir)
                     if f.startswith(".refmod-") or f.endswith(".tmp")]
        self.assertEqual(remaining, [])

    def test_K5_creates_missing_directories(self):
        """Saving into a not-yet-created directory creates it."""
        nested = os.path.join(self.tmpdir, "a", "b", "ref")
        dest = save_refmod(_make_refmod(), nested)
        self.assertTrue(os.path.isfile(dest))


# ═══════════════════════════════════════════════════════════════════════════
# L. MiniMaxH3Mod v4/v5 cross-compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestMiniMaxH3ModCompat(unittest.TestCase):
    """L1–L5: reading files authored by Luisacaotica/ComfyUI-MiniMaxH3Mod."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "legacy")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_L1_legacy_standalone_v4_loads(self):
        """A MiniMaxH3Mod standalone file (key 'latent', mode 'pooled',
        v2 header, no newer fields) loads as a canonical COMPRESSED RefMod."""
        latent = _make_video_latent(t=4, h=8, w=8)
        meta = {"name": "legacy_vid", "kind": "video", "mode": "pooled",
                "latent_t": 4, "latent_h": 8, "latent_w": 8,
                "source": "stack", "pool": "4x8x8", "optimize_steps": 500,
                "tags": ["legacy"], "_format_version": 2}
        _write_raw(self.path, meta, {"latent": latent})
        r = load_refmod(self.path)
        self.assertEqual(r.name, "legacy_vid")
        self.assertEqual(r.kind, "video")
        self.assertEqual(r.mode, "COMPRESSED")
        self.assertEqual((r.latent_t, r.latent_h, r.latent_w), (4, 8, 8))
        self.assertEqual(r.tags, ["legacy"])
        self.assertEqual(r.optimize_steps, 500)
        self.assertEqual(r.description, "")
        self.assertEqual(r.concept_type, "generic")
        self.assertEqual(r.sample_rate, 32000)
        self.assertTrue(torch.equal(r.latent, latent))

    def test_L2_legacy_encode_mode_loads_as_full(self):
        """mode 'encode' (MiniMaxH3Mod current FULL name) loads as FULL."""
        latent = _make_image_latent(h=8, w=8)
        meta = {"name": "enc", "kind": "image", "mode": "encode",
                "latent_t": 1, "latent_h": 8, "latent_w": 8,
                "_format_version": 4}
        _write_raw(self.path, meta, {"latent": latent})
        r = load_refmod(self.path)
        self.assertEqual(r.mode, "FULL")
        self.assertEqual(r.name, "enc")

    def test_L3_missing_name_falls_back_to_basename(self):
        """Metadata without a 'name' falls back to the file basename."""
        meta = {"kind": "image", "mode": "encode", "_format_version": 4,
                "latent_t": 1, "latent_h": 8, "latent_w": 8}
        _write_raw(self.path, meta, {"latent": _make_image_latent()})
        r = load_refmod(self.path)
        self.assertEqual(r.name, os.path.basename(self.path))

    @unittest.skipUnless(os.path.isfile(_REFERENCE_EXAMPLE),
                         "reference artifact not present")
    def test_L4_real_minimax_h3mod_artifact_loads(self):
        """The shipped MiniMaxH3Mod example mod loads end-to-end."""
        raw = read_refmod_meta(os.path.splitext(_REFERENCE_EXAMPLE)[0])
        self.assertEqual(raw["_format_version"], 2)
        r = load_refmod(os.path.splitext(_REFERENCE_EXAMPLE)[0])
        self.assertEqual(r.name, "vanellope_example")
        self.assertEqual(r.kind, "video")
        self.assertEqual(r.mode, "COMPRESSED")
        self.assertEqual(tuple(r.latent.shape[:2]), (1, 24))
        self.assertEqual((r.latent_t, r.latent_h, r.latent_w),
                         tuple(r.latent.shape[2:]))
        self.assertEqual(r.optimize_steps, 500)
        block = convert_to_native_block(r, strength=1.0)
        self.assertEqual(block["kind"], "video")
        self.assertEqual(block["latent_t"], r.latent_t)
        self.assertIs(block["audio_latent"], None)

    def test_L5_legacy_mode_bundle_loads(self):
        """A v5 bundle whose members use legacy mode names loads cleanly."""
        image = _make_refmod(name="i", kind="image")
        audio = _make_refmod(name="a", kind="audio",
                             latent=_make_audio_latent())
        im = image.metadata_dict()
        im["mode"] = "encode"
        am = audio.metadata_dict()
        am["mode"] = "training"
        metadata = {"_format_version": 5, "kind": "bundle", "name": "legacy_b",
                    "members": [im, am]}
        _write_raw(self.path, metadata,
                   {"ref_0": image.latent, "ref_1": audio.latent})
        loaded = load_bundle(self.path, selection="All")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0][0].mode, "FULL")
        self.assertEqual(loaded[0][0].name, "i")
        self.assertEqual(loaded[1][0].mode, "COMPRESSED")
        self.assertEqual(tuple(loaded[1][0].latent.shape), (1, 32, 2, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
