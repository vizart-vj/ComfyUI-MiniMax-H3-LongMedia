"""Focused contracts for the Director-to-native-H3 RefMod seam."""

from dataclasses import replace
import ast
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
pkg = types.ModuleType("ComfyUI-MiniMax-H3-LongMedia")
pkg.__path__ = [str(ROOT)]
pkg.__package__ = "ComfyUI-MiniMax-H3-LongMedia"
for module_name in (
    "ComfyUI-MiniMax-H3-LongMedia",
    "ComfyUI-MiniMax-H3-LongMedia.__init__",
    "__init__",
):
    sys.modules.setdefault(module_name, pkg)

from media_plan import LongMediaPlan


NODES = (ROOT / "nodes.py").read_text(encoding="utf-8")
INIT = (ROOT / "__init__.py").read_text(encoding="utf-8")
DIRECTOR = (ROOT / "director_plan.py").read_text(encoding="utf-8")
DIRECTOR_JS = (ROOT / "web" / "longmedia_director.js").read_text(encoding="utf-8")
FACADE_JS = (ROOT / "web" / "node_facade.js").read_text(encoding="utf-8")
PLANNER_JS = (ROOT / "web" / "longmedia_planner.js").read_text(encoding="utf-8")
CAMERAS_JS = (ROOT / "web" / "longmedia_cameras.js").read_text(encoding="utf-8")


def _load_nodes_function(name):
    tree = ast.parse(NODES)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "nodes.py", "exec"), scope)
    return scope[name]


def _minimal_plan():
    return LongMediaPlan(
        mode="multiclip", duration_basis="frames", total_duration=1.0,
        output_frames=5, segment_frames=5, segment_lengths=(5,),
        segment_starts=(0,), overlap_frames=0, step_frames=5,
        passes=1, generated_frames=5, trim_frames=0,
        resolution_mode="native", video_fps=24.0,
    )


def test_plan_carries_per_segment_refmod_specs():
    assert "segment_refmods" in LongMediaPlan.__dataclass_fields__
    updated = replace(_minimal_plan(), segment_refmods=(({"ref_index": 0},),))
    assert updated.segment_refmods[0][0]["ref_index"] == 0


def test_runtime_exposes_native_ref_descriptors_to_attention_router():
    assert "latentlab_h3_refmod_blocks" in NODES
    assert "payload.get('refs')" in NODES or 'payload.get("refs")' in NODES


def test_attention_path_consumes_refmod_specs_not_only_writes_them():
    marker = "def _run_h3_temporal_embedding_attention("
    start = NODES.index(marker)
    end = NODES.index("\ndef ", start + len(marker))
    body = NODES[start:end]
    assert "latentlab_h3_refmod_specs" in body
    assert "merge_attention_routes" in body


def test_segmented_pass_zero_attachment_does_not_pollute_source_positive():
    setup_start = NODES.index("# Director RefMods are independent")
    setup_end = NODES.index("# v0.6.15", setup_start)
    setup_block = NODES[setup_start:setup_end]
    assert "if _segment_refmod_blocks and _segment_refmod_blocks[0]" not in setup_block
    preencode_start = NODES.index("def _v57_preencode_segment_conditionings")
    preencode_end = NODES.index("\ndef ", preencode_start + 10)
    preencode = NODES[preencode_start:preencode_end]
    assert "raw_result[0]" in preencode
    assert "refmod_blocks[0]" in preencode


def test_refmod_file_identity_uses_full_safe_id_not_eight_char_prefix():
    assert "refmod_id[:8]" not in INIT
    assert "__{refmod_id}" in INIT


def test_cached_selective_segments_do_not_materialize_refmods():
    start = NODES.index("_direct_clips_for_refmods =")
    end = NODES.index("_segment_refmod_blocks =", start)
    assert "_director_required_conditioning_indices" in NODES[start:end]


def test_motion_prompt_source_survives_python_and_js_normalization():
    assert '"motion_prompt_source"' in DIRECTOR
    shot_start = DIRECTOR_JS.index("function normalizeShot")
    shot_end = DIRECTOR_JS.index("\nfunction ", shot_start + 5)
    assert "motion_prompt_source" in DIRECTOR_JS[shot_start:shot_end]


def test_frontend_refmod_catalog_limit_matches_python_limit():
    assert "const REFMOD_MAX = 256;" in DIRECTOR_JS


def test_all_editor_dom_widgets_are_explicitly_presentation_only():
    for source, widget_name in (
        (FACADE_JS, "multiclip_editor"),
        (PLANNER_JS, "clip_editor"),
        (CAMERAS_JS, "camera_editor"),
    ):
        start = source.index(f'addDOMWidget("{widget_name}"')
        end = source.index("computeSize", start)
        block = source[start:end]
        assert "__lmPresentationOnly" in block
        assert "serializeValue" in block


def test_refmod_attention_dispatches_through_comfy_selected_masked_backend():
    start = NODES.index("def _run_h3_temporal_embedding_attention(")
    end = NODES.index("def _h3_existing_calibrate_runtime_budget(", start)
    block = NODES[start:end]
    assert "from comfy.ldm.modules.attention import optimized_attention" in block
    assert "part = optimized_attention(" in block
    assert "mask=mask, skip_reshape=True" in block
    assert "part = attention_pytorch(" not in block


def test_inactive_picture_is_available_to_refmod_without_becoming_normal_image_input():
    subject_id = "subject-inactive-refmod"
    pixels = object()
    builder = _load_nodes_function("_lm_build_director_media_bundle")
    builder.__globals__.update({
        "_director_normalize_document": lambda value: value,
        "_director_picture_runtime_map": lambda document: {
            "runtime_slots": {},
            "anchors": {
                "anchor_subject_ids": [],
                "first_subject_id": None,
                "last_subject_id": None,
            },
        },
        "_director_frame_anchor_contract": lambda document: {
            "anchor_subject_ids": [],
            "first_subject_id": None,
            "last_subject_id": None,
        },
        "_lm_director_usage_ranges": _load_nodes_function("_lm_director_usage_ranges"),
        "_lm_director_unique_trim": lambda ranges: None,
        "_lm_director_media_path": lambda record: record["filename"],
        "_lm_call_core_node": lambda name, **kwargs: (pixels,),
    })
    resolver = _load_nodes_function("_lm_refmod_source_payload")
    resolver.__globals__["torch"] = types.SimpleNamespace(
        is_tensor=lambda value: value is pixels,
    )
    document = {
        "subjects": [{
            "subject_id": subject_id,
            "name": "Inactive test picture",
            "kind": "Picture",
            "slot": 1,
            "media": {"filename": "test.png", "subfolder": ""},
        }],
        "shots": [],
        "audio_blocks": [],
        "extra_tracks": [],
        "refmods": [{
            "source": f"subject:{subject_id}",
            "source_subject_id": subject_id,
        }],
    }

    bundle = builder(document)
    payload = resolver(document["refmods"][0], document, bundle, "test-project")

    assert payload["image"] is pixels
    assert bundle["images"][0] is None
    assert bundle["manifest"][0]["loaded_as"] == "refmod_source"


def test_hires_refine_keeps_outside_refmod_as_inactive_descriptor():
    slicer = _load_nodes_function("_lm_slice_refmod_specs_for_hires_refine")
    specs = (
        {"refmod_id": "outside", "ref_index": 2, "start_frame": 0, "end_frame": 10},
        {"refmod_id": "inside", "ref_index": 3, "start_frame": 25, "end_frame": 40},
    )
    localized = slicer(specs, window_start_frame=20, window_frame_count=10)
    assert len(localized) == 2
    assert (localized[0]["start_frame"], localized[0]["end_frame"]) == (0, 0)
    assert (localized[1]["start_frame"], localized[1]["end_frame"]) == (5, 10)


def test_motion_repair_retimes_every_refmod_descriptor_including_inactive():
    slicer = _load_nodes_function("_lm_slice_refmod_specs_for_motion_repair")
    dilation = {"expanded_frame_to_source_frame": (0, 0, 1, 2, 2)}
    specs = (
        {"refmod_id": "active", "ref_index": 2, "start_frame": 11, "end_frame": 12},
        {"refmod_id": "inactive", "ref_index": 3, "start_frame": 20, "end_frame": 21},
    )
    localized = slicer(specs, dilation, source_frame_offset=10)
    assert len(localized) == 2
    assert (localized[0]["start_frame"], localized[0]["end_frame"]) == (2, 3)
    assert (localized[1]["start_frame"], localized[1]["end_frame"]) == (0, 0)


def test_motion_repair_installs_and_restores_local_refmod_specs():
    start = NODES.index("_mr_transformer_options = None")
    end = NODES.index("if not getattr(_mr_candidate", start)
    block = NODES[start:end]
    assert "_lm_slice_refmod_specs_for_motion_repair" in block
    assert "_mr_prev_refmod_specs" in block
    assert "latentlab_h3_refmod_single_pass" in block


def test_attention_validates_authored_target_clock_against_native_lattice():
    localize_start = NODES.index("def _lm_localize_refmod_specs")
    localize_end = NODES.index("\ndef ", localize_start + 5)
    assert "target_frame_count" in NODES[localize_start:localize_end]
    attention_start = NODES.index("def _run_h3_temporal_embedding_attention(")
    attention_end = NODES.index("\ndef ", attention_start + 5)
    body = NODES[attention_start:attention_end]
    assert "expected_frame_count=" in body


# -----------------------------------------------------------------------------
# v0.6.57: Director ENCODE is VAE-only, and Setup refuses unprepared RefMods
# instead of silently running the encoder on the render path.
# -----------------------------------------------------------------------------

def _nodes_function_body(name: str) -> str:
    """Exact source of one top-level ``nodes.py`` function."""
    start = NODES.index(f'def {name}(')
    end = NODES.find('\ndef ', start + 1)
    return NODES[start:end if end != -1 else None]


def test_director_encode_surface_is_vae_only():
    body = _nodes_function_body('_lm_refmod_encode_one')
    loader = _nodes_function_body('_lm_load_h3_vae')
    assert 'import comfy.sd' in loader
    assert 'comfy.sd.VAE(' in loader
    for forbidden in (
        'load_checkpoint', 'load_diffusion_model', 'ModelPatcher', 'VAELoader',
        'hybrid_sampler', 'hunyuan_sampler', 'KSampler', 'encode_prompt',
    ):
        assert forbidden not in loader
        assert forbidden not in body


def test_setup_refuses_unprepared_refmod_instead_of_encoding():
    body = _nodes_function_body('_lm_refmod_blocks_for_segment')
    assert '_lm_refmod_encode_record' not in body
    assert 'save_bundle' not in body
    assert 'is not encoded yet' in body
    assert 'press ENCODE' in body


def test_refmod_native_descriptor_rides_the_authored_route():
    body = _nodes_function_body('_lm_refmod_blocks_for_segment')
    assert "'native_block': dict(block)" in body


def test_attention_recovers_refmod_descriptors_through_a_layered_fallback():
    attention = _nodes_function_body('_run_h3_temporal_embedding_attention')
    assert '_lm_h3_recover_refmod_descriptors(transformer_options)' in attention
    assert '_lm_refmod_blocks_for_segment' not in attention
    # The cached route key must stay cheap: live tensors never enter it.
    assert "if k != 'native_block'" in attention


def test_dangling_refmod_reference_fails_loudly_and_packing_is_reported():
    """Regression: a RefMod render that silently applies nothing.

    ``_lm_refmod_blocks_for_segment`` used to ``continue`` past a span whose
    ``refmod_id`` was no longer in the document, so a clip left dangling by a
    SYNC removal produced a completely normal render with no RefMod effect and
    no error. Setup now refuses and names the reference, and what actually got
    packed is printed so "is it applied?" has a console answer.
    """
    body = _nodes_function_body('_lm_refmod_blocks_for_segment')
    # NB: asserted against SOURCE text, so split literals need split fragments.
    assert 'but that RefMod is not in the ' in body
    assert 'Director document. A removed RefMod leaves its timeline clip dangling.' in body
    assert 're-import that RefMod or' in body
    # the silent drop must be gone
    assert 'if not isinstance(record, dict):\n            continue' not in body
    assert '[DIRECTOR REFMOD] packed' in body
    # strength is reported per packed block, not just accepted
    assert 'str {strength:.2f}' in body


def test_lost_minimax_refs_are_reinjected_into_the_h3_payload():
    guard = _nodes_function_body('_h3_segment_layout_guard_wrapper')
    assert '_lm_h3_reinject_refmod_refs' in guard
    assert '_lm_h3_reinject_refmod_refs' in NODES


def test_refmod_descriptors_are_reinjected_when_payload_refs_are_lost():
    recover = _load_nodes_function("_lm_h3_recover_refmod_descriptors")
    reinject = _load_nodes_function("_lm_h3_reinject_refmod_refs")
    reinject.__globals__["_lm_h3_recover_refmod_descriptors"] = recover
    reinject.__globals__["_lm_print"] = lambda *args, **kwargs: None
    block = {
        "kind": "image", "latent_h": 8, "latent_w": 8, "latent": "IMG",
        "longmedia_refmod_id": "rm1", "longmedia_refmod_member_index": 0,
        "longmedia_refmod_appended_block_index": 0,
    }
    specs = (
        {"refmod_id": "rm1", "member_index": 0, "appended_block_index": 0,
         "native_block": block},
    )
    options = {"latentlab_h3_refmod_specs": specs, "cond_or_uncond": [0]}
    payload = {"layout": object(), "cond_video_latents": []}
    out, changed = reinject(payload, options, 8, None)
    assert changed is True
    assert out["refs"][0]["longmedia_refmod_id"] == "rm1"
    assert out["cond_video_latents"] == ["IMG"]
    assert out["longmedia_refmod_recovered_descriptors"] == 1
    # The stale layout must not survive: H3 repacks the exact ref rows.
    assert "layout" not in out
    assert payload.get("refs") is None

    # Negative CFG never receives positive-only native refs.
    negative = {"latentlab_h3_refmod_specs": specs, "cond_or_uncond": [1]}
    out_negative, changed_negative = reinject({"layout": object()}, negative, 8, None)
    assert changed_negative is False
    assert "refs" not in out_negative

    # Batched CFG keeps its own unbatched-branch contract error.
    batched = {"latentlab_h3_refmod_specs": specs, "cond_or_uncond": [0, 1]}
    _out_batched, changed_batched = reinject({"layout": object()}, batched, 8, None)
    assert changed_batched is False

    # Already-attached refs are never duplicated.
    out_ready, changed_ready = reinject(
        {"refs": [{"kind": "image"}]}, options, 8, None,
    )
    assert changed_ready is False
    assert out_ready["refs"] == [{"kind": "image"}]


def test_native_blocks_land_on_the_h3_patch_grid():
    """Regression: ``shape '[1, 24, 1, 1, 30, 2, 20, 2]' is invalid for input of
    size 59040``.

    Host ``patchify_video`` floor-divides the latent H/W by the 2x2 spatial patch
    and reshapes to exactly ``h*ph`` by ``w*pw`` columns, so an odd latent width
    loses a column and the reshape dies deep inside the host forward. RefMod
    sources are arbitrary user media, so the descriptor must hand the host a
    latent that already sits on the patch grid.
    """
    import torch
    from refmod_backend import RefMod, convert_to_native_block

    def host_patchify(latent, patch_size=(1, 2, 2)):
        # verbatim arithmetic of comfy/ldm/minimax/model.py:42-49
        b, c, t_full, h_full, w_full = latent.shape
        pt, ph, pw = patch_size
        t, h, w = t_full // pt, h_full // ph, w_full // pw
        return latent.reshape(b, c, t, pt, h, ph, w, pw)

    # (1,24,1,60,41) is the exact shape from the reported failure.
    for shape in ((1, 24, 1, 121, 82), (1, 24, 1, 60, 41), (1, 24, 3, 61, 40)):
        latent = torch.zeros(shape)
        kind = "video" if shape[2] > 1 else "image"
        block = convert_to_native_block(
            RefMod(name="t", kind=kind, latent=latent, mode="FULL")
        )
        assert block["latent_h"] % 2 == 0, shape
        assert block["latent_w"] % 2 == 0, shape
        assert block["latent"].shape[-2:] == (block["latent_h"], block["latent_w"])
        host_patchify(block["latent"])  # must not raise

    # The trailing odd column is what gets dropped.
    odd = torch.zeros(1, 24, 1, 4, 5)
    odd[..., :, 4] = 9.0
    fitted = convert_to_native_block(
        RefMod(name="t", kind="image", latent=odd, mode="FULL")
    )["latent"]
    assert fitted.shape == (1, 24, 1, 4, 4)
    assert float(fitted.sum()) == 0.0

    # Already-aligned latents pass through untouched.
    clean = torch.zeros(1, 24, 1, 8, 6)
    out = convert_to_native_block(
        RefMod(name="t", kind="image", latent=clean, mode="FULL")
    )
    assert out["latent"] is clean
    assert (out["latent_h"], out["latent_w"]) == (8, 6)

    # Audio keeps its own [1, 32, 2, T] contract untouched.
    audio = torch.zeros(1, 32, 2, 7)
    ref = RefMod(name="a", kind="audio", latent=audio, mode="FULL")
    assert convert_to_native_block(ref)["audio_latent"] is audio


def test_refmod_identity_survives_the_native_hop_that_strips_it():
    """Regression: ``Director RefMod descriptor mapping failed``.

    ``convert_to_native_block`` emits only the native descriptor schema and host
    ``PackedLayout(refs=...)`` contracts to that schema alone, so the
    ``longmedia_refmod_*`` annotations are dropped anywhere on the CONDITIONING /
    ``payload['refs']`` / layout path. Recovery must re-stamp them from the
    authored specs -- including when the host's own native refs precede ours.
    """
    recover = _load_nodes_function("_lm_h3_recover_refmod_descriptors")
    from refmod_routing import resolve_refmod_spec_index as resolve

    stripped = {"kind": "image", "latent_h": 8, "latent_w": 8, "latent": "IMG"}
    authored = dict(
        stripped,
        longmedia_refmod_id="refmod-a88e",
        longmedia_refmod_member_index=0,
        longmedia_refmod_appended_block_index=0,
        longmedia_refmod_strength=1.0,
    )
    specs = (
        {"refmod_id": "refmod-a88e", "member_index": 0, "appended_block_index": 0,
         "native_block": authored},
    )
    spec = dict(specs[0])

    # The side channel / layout carrier arrives with identity stripped.
    got = recover({"latentlab_h3_refmod_blocks": (dict(stripped),),
                   "latentlab_h3_refmod_specs": specs})
    assert [block["longmedia_refmod_id"] for block in got] == ["refmod-a88e"]
    assert resolve(got, spec) == 0

    # The host's own native refs precede ours (ref_index = native + appended).
    native_ref = {"kind": "image", "latent_h": 4, "latent_w": 4, "latent": "OTHER"}
    mixed = recover({"latentlab_h3_refmod_blocks": (dict(native_ref), dict(stripped)),
                     "latentlab_h3_refmod_specs": specs})
    assert len(mixed) == 2
    assert "longmedia_refmod_id" not in mixed[0]
    assert mixed[1]["longmedia_refmod_id"] == "refmod-a88e"
    assert resolve(mixed, spec) == 1

    # Only the authored specs left still resolves to the right descriptor.
    assert resolve(recover({"latentlab_h3_refmod_specs": specs}), spec) == 0


def test_refmod_identity_restamping_leaves_no_identity_less_block_behind():
    recover = _load_nodes_function("_lm_h3_recover_refmod_descriptors")
    stripped = {"kind": "audio", "ref_audio_t": 2, "audio_latent": "A"}
    specs = tuple(
        {"refmod_id": f"rm{index}", "member_index": 0, "appended_block_index": index,
         "native_block": dict(stripped, longmedia_refmod_id=f"rm{index}",
                              longmedia_refmod_member_index=0,
                              longmedia_refmod_appended_block_index=index,
                              longmedia_refmod_strength=0.5)}
        for index in range(3)
    )
    got = recover({"latentlab_h3_refmod_blocks": tuple(dict(b) for b in (stripped,) * 3),
                   "latentlab_h3_refmod_specs": specs})
    assert [block["longmedia_refmod_id"] for block in got] == ["rm0", "rm1", "rm2"]
    assert [block["longmedia_refmod_appended_block_index"] for block in got] == [0, 1, 2]
    assert all(block["longmedia_refmod_strength"] == 0.5 for block in got)


def test_multiclip_refmod_descriptors_are_reordered_after_continuation_refs():
    order = _load_nodes_function("_lm_order_refmod_refs_last")
    order.__globals__["_conditioning_meta"] = lambda entry: (
        entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1
        and isinstance(entry[1], dict) else entry if isinstance(entry, dict) else None
    )
    refmod = {
        "kind": "image", "longmedia_refmod_id": "rm-a",
        "longmedia_refmod_member_index": 0,
        "longmedia_refmod_appended_block_index": 0,
    }
    continuation = {"kind": "video_audio", "longmedia_native_av_context": True}
    positive = [["COND", {"minimax_refs": [refmod, continuation]}]]
    result = order(positive, ({"refmod_id": "rm-a", "member_index": 0,
                              "appended_block_index": 0},))
    assert result[0][1]["minimax_refs"] == [continuation, refmod]


def test_segmented_refmod_selection_uses_workflow_not_executor_mode():
    setup_start = NODES.index("# Director RefMods are independent")
    setup_end = NODES.index("# v0.6.15", setup_start)
    setup_block = NODES[setup_start:setup_end]
    assert "if workflow_mode == 'multiclip'" in setup_block
    assert "if getattr(plan, 'mode', None) == 'multiclip' and _seg_idx < len(multiclip_clips)" not in setup_block


def test_refmod_descriptors_recover_from_layout_carrier_and_specs():
    recover = _load_nodes_function("_lm_h3_recover_refmod_descriptors")
    block = {"kind": "audio", "longmedia_refmod_appended_block_index": 0}
    assert recover({"latentlab_h3_refmod_blocks": (block,)}) == (block,)
    carrier = types.SimpleNamespace()
    carrier.ref_blocks = (block,)
    assert recover({"latentlab_h3_packed_layout": carrier}) == (block,)
    assert recover({"latentlab_h3_refmod_specs": ({"native_block": block},)}) == (block,)
    assert recover({}) == ()
    assert recover(None) == ()
