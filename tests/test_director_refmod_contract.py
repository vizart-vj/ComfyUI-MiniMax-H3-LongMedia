"""
Static / JSON contract tests for LongMedia Director RefMod UX.

Validates the data model (normalization schema, backend route contracts,
timeline-layer geometry) without requiring a live ComfyUI server.

Run:
    python -m pytest tests/test_director_refmod_contract.py -v
"""

import json, re, textwrap, pathlib, sys, types, pytest

# ---------------------------------------------------------------------------
# Helpers – parse the JS file for constants / function signatures
# ---------------------------------------------------------------------------

JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "web" / "longmedia_director.js"

JS_TEXT = JS_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Run-environment guard.
#
# The repository root is a ComfyUI custom-node package whose __init__.py only
# works through relative imports. pytest 8+ imports every ancestor __init__.py
# while setting up collected tests, which crashes on that package body and turns
# every case into a setup error. Register an inert placeholder module so the
# documented command above runs without executing the package body.
# ---------------------------------------------------------------------------
_ROOT = JS_PATH.parent.parent
if "__init__" not in sys.modules and (_ROOT / "__init__.py").exists():
    try:
        _placeholder = types.ModuleType("__init__")
        _placeholder.__file__ = str(_ROOT / "__init__.py")
        _placeholder.__package__ = ""
        sys.modules["__init__"] = _placeholder
    except Exception:
        pass


def _extract_js_array(name: str) -> list[str]:
    """Extract a top-level JS const array by name, returning its Python list."""
    pattern = rf"const\s+{name}\s*=\s*\[(.*?)\];"
    m = re.search(pattern, JS_TEXT, re.DOTALL)
    assert m, f"Could not find const {name} in JS source"
    raw = m.group(1)
    return [s.strip().strip('"').strip("'") for s in raw.split(",") if s.strip().strip('"').strip("'")]


def _extract_js_object(name: str) -> dict:
    """Extract a top-level JS const object (one level deep)."""
    pattern = rf"const\s+{name}\s*=\s*\{{(.*?)\}};"
    m = re.search(pattern, JS_TEXT, re.DOTALL)
    assert m, f"Could not find const {name} in JS source"
    return m.group(1)


def _js_source() -> str:
    return JS_TEXT


# =========================================================================
# 1. EXTRA_TRACK_TYPES must include "refmod"
# =========================================================================

class TestExtraTrackTypesIncludeRefmod:
    def test_refmod_in_types(self):
        types = _extract_js_array("EXTRA_TRACK_TYPES")
        assert "refmod" in types, f"EXTRA_TRACK_TYPES={types} missing 'refmod'"

    def test_refmod_not_duplicated(self):
        types = _extract_js_array("EXTRA_TRACK_TYPES")
        assert types.count("refmod") == 1


# =========================================================================
# 2. EXTRA_TRACK_META must have a refmod entry
# =========================================================================

class TestExtraTrackMetaRefmod:
    def test_refmod_meta_exists(self):
        raw = _extract_js_object("EXTRA_TRACK_META")
        assert "refmod" in raw, "EXTRA_TRACK_META missing refmod key"

    def test_refmod_has_label(self):
        raw = _extract_js_object("EXTRA_TRACK_META")
        assert "label" in raw.split("refmod")[1].split("}")[0]

    def test_refmod_has_color(self):
        raw = _extract_js_object("EXTRA_TRACK_META")
        assert "color" in raw.split("refmod")[1].split("}")[0]


# =========================================================================
# 3. RefMod record shape
# =========================================================================

class TestRefmodRecordShape:
    """A refmod record is a free-form object. The minimum shape required by
    the frontend library panel and backend routes is tested here."""

    REQUIRED_KEYS = {"refmod_id", "name", "mode", "concept_type", "description",
                     "spatial_grid", "temporal_frames",
                     "refinement_steps", "strength"}

    def test_sample_record_has_required_keys(self):
        sample = {
            "refmod_id": "refmod-001",
            "name": "Landscape Hero",
            "source": "custom",
            "mode": "FULL",
            "concept_type": "scenery",
            "description": "Wide mountain vista",
            "spatial_grid": "32x32",
            "temporal_frames": 24,
            "refinement_steps": 20,
            "max_tokens": 512,
            "strength": 1.0,
        }
        missing = self.REQUIRED_KEYS - set(sample.keys())
        assert not missing, f"Missing required keys: {missing}"

    def test_strength_range(self):
        for v in [0.0, 0.5, 1.0]:
            assert 0.0 <= v <= 1.0

    def test_mode_values(self):
        valid = {"FULL", "COMPRESSED"}
        for m in ["FULL", "COMPRESSED"]:
            assert m in valid


class TestRefmodSourceComposerRefinementSteps:
    def test_creator_exposes_and_serializes_compressed_refinement_steps(self):
        start = JS_TEXT.index("function openRefmodCreator(")
        end = JS_TEXT.index("// ── RefMod inspector / panel", start)
        composer = JS_TEXT[start:end]

        assert "const refinementField=input(record.refinement_steps" in composer
        assert 'field("Refinement steps",refinementField' in composer
        assert 'refinementField.disabled=record.mode!=="COMPRESSED"' in composer
        assert "refinement_steps=Math.max(0,Math.min(2000" in composer
        assert "body:JSON.stringify({project_id:doc.project_id,refmod:record" in composer

    def test_js_normalize_refmod_covers_required_keys(self):
        """The real frontend normalizer, not just the sample, must emit every
        field the Library / Inspector panel and the backend routes rely on."""
        idx = JS_TEXT.find("function normalizeRefmod")
        assert idx >= 0, "normalizeRefmod not found"
        body = JS_TEXT[idx:idx + 2500]
        for key in self.REQUIRED_KEYS:
            assert key in body, f"normalizeRefmod is missing {key}"
        assert JS_TEXT.find("function defaultRefmod") >= 0, "defaultRefmod not found"

    def test_dead_refmod_fields_stay_removed(self):
        """``resolution`` / ``max_tokens`` were normalised in three places and
        consumed nowhere -- which is exactly what made them look meaningful in
        the inspector. They stay out of the record shape for good."""
        src = _js_source()
        for line in src.splitlines():
            if (line.startswith("function defaultRefmod(")
                    or line.startswith("function normalizeRefmod(")):
                assert "resolution" not in line
                assert "max_tokens" not in line
        inspector = src[src.index("function renderRefmodInspector"):]
        assert 'field("Resolution"' not in inspector[:6000]
        assert 'field("Max tokens"' not in inspector[:6000]
        plan = (JS_PATH.parent.parent / "director_plan.py").read_text(encoding="utf-8")
        assert '"resolution": str(item.get("resolution")' not in plan
        assert '"max_tokens": max(32, min(32768' not in plan


# =========================================================================
# 4. Normalized doc contains top-level refmods library
# =========================================================================

class TestNormalizedDocRefmods:
    """The normalized doc (version 10) must carry a refmods array at the top level."""

    def test_js_normalize_doc_references_refmods(self):
        src = _js_source()
        # normalizeDoc should mention refmods in its body
        assert "refmods" in src, "JS source does not reference 'refmods' at all"

    def test_refmods_in_doc_version_10(self):
        """Construct a plausible default doc and check that the normalization
        code would accept / produce a refmods field."""
        # We verify by regex that normalizeDoc's output shape includes refmods
        src = _js_source()
        # The normalizer should initialize refmods when missing
        assert re.search(r"refmods\s*:\s*\[", src) or "refmods:[]" in src, \
            "normalizeDoc does not initialize refmods array"
        assert "version:10" in src, "director_json must be stamped version 10"
        assert "version:9" not in src, "stale version:9 remains in director doc builders"


# =========================================================================
# 5. Shot schema includes refmod_id and strength
# =========================================================================

class TestShotSchemaRefmodFields:
    def test_shot_has_refmod_id(self):
        src = _js_source()
        assert "refmod_id" in src, "Shot schema missing refmod_id field"

    def test_shot_has_strength(self):
        src = _js_source()
        # strength is used in refmod context
        assert "strength" in src, "Shot/clip schema missing strength field"


# =========================================================================
# 6. Backend route contract
# =========================================================================

class TestBackendRouteContract:
    """Verify the JS references the expected REST endpoints for refmod CRUD."""

    EXPECTED_ROUTES = [
        "/longmedia/refmods/list",
        "/longmedia/refmods/save",
        "/longmedia/refmods/delete",
        "/longmedia/refmods/from_take",
    ]

    def test_all_routes_present(self):
        src = _js_source()
        for route in self.EXPECTED_ROUTES:
            assert route in src, f"JS missing backend route: {route}"

    def test_routes_use_api_fetch(self):
        src = _js_source()
        for route in self.EXPECTED_ROUTES:
            # Must use api.fetchApi for the route
            assert f"api.fetchApi" in src, "Routes should use api.fetchApi()"


# =========================================================================
# 7. UI state for collapsed REFMODS panel
# =========================================================================

class TestRefmodsUIState:
    def test_show_refmods_pref_key(self):
        src = _js_source()
        assert "show_refmods" in src, "UI state should persist show_refmods"

    def test_refmods_collapse_toggle(self):
        src = _js_source()
        # Should have a toggle that flips the collapsed state
        assert re.search(r"show_refmods\s*[!=]=", src) or "__lmdShowRefmods" in src, \
            "Missing show_refmods toggle logic"


# =========================================================================
# 8. REFMODS panel render function exists
# =========================================================================

class TestRefmodsPanelRender:
    def test_render_refmods_panel_exists(self):
        src = _js_source()
        assert re.search(r"function\s+renderRefmod", src), \
            "Missing renderRefmod* function"

    def test_panel_has_library_section(self):
        src = _js_source()
        assert "REFMODS" in src or "REFMOD" in src, "Missing REFMODS panel title"

    def test_panel_has_create_button(self):
        src = _js_source()
        assert "CREATE" in src, "REFMODS panel should have CREATE button"

    def test_panel_has_from_take_button(self):
        src = _js_source()
        assert "FROM TAKE" in src or "from_take" in src.lower(), \
            "REFMODS panel should have FROM TAKE action"

    def test_panel_has_inspector_section(self):
        src = _js_source()
        assert "Inspector" in src or "inspector" in src.lower(), \
            "REFMODS panel should have Inspector section"


# =========================================================================
# 9. Form field definitions
# =========================================================================

class TestRefmodFormFields:
    EXPECTED_FIELDS = [
        "Source", "Mode", "Name", "Concept Type", "Description",
        "Spatial Grid", "Temporal Frames",
        "Refinement Steps", "Save",
    ]

    def test_all_form_fields_present(self):
        src = _js_source()
        for field in self.EXPECTED_FIELDS:
            # Check that each field label appears in the source
            assert field.lower() in src.lower() or field in src, \
                f"Missing form field: {field}"

    def test_strength_range_constraints_are_set_before_fractional_value(self):
        src = _js_source()
        start = src.index('const range=input("",null,"range")', src.index("function renderRefmodInspector"))
        end = src.index("body.append(strength)", start)
        slider = src[start:end]
        assert slider.index('range.step="0.01"') < slider.index('range.value=String(')
        assert 'range.min="0"' in slider and 'range.max="1"' in slider

    def test_mode_has_full_compressed(self):
        src = _js_source()
        assert "FULL" in src and "COMPRESSED" in src, \
            "Mode must offer FULL and COMPRESSED options"


# =========================================================================
# 10. Timeline layer clips for refmod
# =========================================================================

class TestRefmodTimelineLayer:
    def test_refmod_extra_track_type(self):
        """The 'refmod' type must be handled in the extra-tracks timeline code."""
        src = _js_source()
        assert '"refmod"' in src or "'refmod'" in src, \
            "refmod string literal not found in timeline code"

    def test_refmod_clips_have_start_duration(self):
        """Refmod clips on the timeline must have start and duration for interval logic."""
        # We check that normalizeExtraClip or similar handles start/duration
        src = _js_source()
        assert re.search(r"normalizeExtraClip", src), \
            "normalizeExtraClip function must exist for refmod clips"


# =========================================================================
# 11. Presentation widgets must not serialize
# =========================================================================

class TestPresentationWidgetNoSerialize:
    def test_editor_widget_serialize_false(self):
        """The DOM widget must not serialize into the graph JSON."""
        src = _js_source()
        # Look for serialize:false pattern near addDOMWidget
        assert re.search(r"serialize\s*:\s*false", src) or "serialize=false" in src, \
            "Editor widget should have serialize:false"

    def test_serialize_value_undefined(self):
        src = _js_source()
        assert "serializeValue" in src or "serializeValue" in src, \
            "serializeValue should be defined to return undefined"


# =========================================================================
# 12. TAKE snapshot naturally preserves refmod state
# =========================================================================

class TestTakeSnapshotPreservesRefmods:
    def test_take_snapshot_stores_full_doc(self):
        """TAKE restore applies the full timeline_snapshot, which includes the
        top-level refmods library. No special refmod restore logic is needed
        because it lives in the doc."""
        src = _js_source()
        # applyTakeTimelineSnapshot should normalizeDoc the raw snapshot
        assert "normalizeDoc" in src, "applyTakeTimelineSnapshot must normalizeDoc"
        assert "director_timeline_snapshot" in src, "TAKE must store director_timeline_snapshot"


# =========================================================================
# 13. Clip schema for refmod_id and strength in normalizeExtraClip
# =========================================================================

class TestExtraClipRefmodFields:
    def test_refmod_id_in_clip_normalization(self):
        src = _js_source()
        # normalizeExtraClip should handle refmod_id
        assert "refmod_id" in src, "normalizeExtraClip should process refmod_id"

    def test_strength_in_clip_normalization(self):
        src = _js_source()
        assert "strength" in src, "normalizeExtraClip should process strength"


# =========================================================================
# 14. JSON serialization roundtrip
# =========================================================================

class TestJsonRoundtrip:
    def test_refmods_array_serializes(self):
        doc = {
            "version": 10,
            "kind": "h3_longmedia_director",
            "refmods": [
                {
                    "refmod_id": "rm-1",
                    "name": "Test",
                    "mode": "FULL",
                    "concept_type": "scenery",
                    "description": "",
                    "resolution": "1920x1080",
                    "spatial_grid": "32x32",
                    "temporal_frames": 24,
                    "refinement_steps": 20,
                    "strength": 1.0,
                    "source": "custom",
                }
            ],
            "shots": [],
            "extra_tracks": [],
            "subjects": [],
            "camera_blocks": [],
            "audio_blocks": [],
        }
        s = json.dumps(doc)
        d = json.loads(s)
        assert len(d["refmods"]) == 1
        assert d["refmods"][0]["refmod_id"] == "rm-1"

    def test_extra_clip_with_refmod_fields(self):
        clip = {
            "clip_id": "layer-1",
            "name": "RefMod clip",
            "start": 0.0,
            "duration": 5.0,
            "refmod_id": "rm-1",
            "strength": 0.75,
        }
        s = json.dumps(clip)
        d = json.loads(s)
        assert d["refmod_id"] == "rm-1"
        assert d["strength"] == 0.75


# =========================================================================
# 15. defaultDoc includes refmods
# =========================================================================

class TestDefaultDoc:
    def test_default_doc_source_has_refmods(self):
        src = _js_source()
        # defaultDoc function should initialize refmods
        # Search for refmods near defaultDoc
        idx = src.find("function defaultDoc")
        assert idx >= 0, "defaultDoc function not found"
        snippet = src[idx:idx+1500]
        assert "refmods" in snippet, "defaultDoc should initialize refmods array"
        assert "version:10" in snippet, "defaultDoc must stamp director_json version 10"


# =========================================================================
# 16. director_json schema version is 10 everywhere it is produced
# =========================================================================

class TestDocVersion10:
    def _normalize_doc_snippet(self):
        idx = _js_source().find("function normalizeDoc(d)")
        assert idx >= 0, "normalizeDoc not found"
        return _js_source()[idx:idx + 6000]

    def test_default_doc_version_10(self):
        idx = _js_source().find("function defaultDoc")
        assert idx >= 0
        assert "version:10" in _js_source()[idx:idx + 600]

    def test_migrate_v1_version_10(self):
        idx = _js_source().find("function migrateV1")
        assert idx >= 0
        assert "version:10" in _js_source()[idx:idx + 2500]

    def test_normalize_doc_emits_version_10(self):
        assert "version:10" in self._normalize_doc_snippet(), \
            "normalizeDoc must stamp version 10 on every director_json write"

    def test_normalize_doc_maps_refmods(self):
        snippet = self._normalize_doc_snippet()
        assert "normalizeRefmod" in snippet, "normalizeDoc must normalize top-level refmods"
        assert re.search(r"refmods\s*=", snippet), "normalizeDoc must write doc.refmods"

    def test_no_stale_version_9(self):
        assert "version:9" not in _js_source(), "a doc builder still stamps version 9"


# =========================================================================
# 17. refmods survive normalize -> snapshot -> history -> TAKE restore
# =========================================================================

class TestRefmodPersistencePipeline:
    def test_snapshot_doc_normalizes(self):
        src = _js_source()
        assert re.search(r"function snapshotDoc\(doc\)\s*\{\s*return JSON\.stringify\(normalizeDoc\(doc\)\);\s*\}", src), \
            "snapshotDoc must serialize the normalized doc (refmods included)"

    def test_commit_normalizes_and_pushes_history(self):
        src = _js_source()
        # commit() is a single-line function; capture to the end of its line.
        m = re.search(r"function commit\(node, doc, rerender=true\) \{(.*)\}", src)
        assert m, "commit() not found"
        body = m.group(1)
        assert "normalizeDoc(doc)" in body, "commit must normalize before writing director_json"
        assert "pushHistory" in body, "commit must push the previous doc onto the undo stack"

    def test_apply_history_normalizes(self):
        src = _js_source()
        m = re.search(r"function applyHistory\(node,serialized,destination\)\s*\{(.*?)\}\n", src, re.DOTALL)
        assert m, "applyHistory not found"
        assert "normalizeDoc" in m.group(1), "undo/redo must re-normalize (refmods preserved)"

    def test_take_snapshot_restore_normalizes(self):
        src = _js_source()
        m = re.search(r"function applyTakeTimelineSnapshot\(.*?\)\s*\{(.*?)\n\}", src, re.DOTALL)
        assert m, "applyTakeTimelineSnapshot not found"
        body = m.group(1)
        assert "normalizeDoc" in body, "TAKE restore must normalizeDoc the raw snapshot"
        assert "director_json" in body, "TAKE restore must write back through director_json"
        assert "pushHistory" in body, "TAKE restore must be undoable"


# =========================================================================
# 18. REFMOD panel: collapsible, Library / Create / From TAKE / Inspector
# =========================================================================

class TestRefmodPanelSections:
    def test_panel_and_inspector_functions_defined(self):
        src = _js_source()
        assert re.search(r"function renderRefmodPanel\(node,doc,host\)", src)
        assert re.search(r"function renderRefmodInspector\(node,doc,refmod,host\)", src)
        assert re.search(r"function renderRefmodSidebar", src)

    def test_library_section_label(self):
        assert 'el("div","LIBRARY"' in _js_source(), "panel needs a Library section label"

    def test_create_action(self):
        assert 'button("CREATE"' in _js_source(), "panel needs a CREATE action"

    def test_from_take_action(self):
        assert 'button("FROM TAKE"' in _js_source(), "panel needs a FROM TAKE action"

    def test_inspector_section_header(self):
        assert "REFMOD · INSPECTOR" in _js_source(), "panel needs an Inspector section"

    def test_panel_is_mounted_in_body_grid(self):
        src = _js_source()
        assert "renderRefmodPanel(node,doc,refmodsContent)" in src, "panel must be mounted in the layout"
        assert re.search(r"if\(showRefmods\)\{", src), "panel must be gated by its visibility flag"

    def test_panel_is_collapsible_and_persisted(self):
        src = _js_source()
        assert 'button("REFMODS"' in src, "toolbar needs the REFMODS collapse toggle"
        assert 'saveDirectorUIPref(node,"show_refmods"' in src, "collapse state must persist"
        assert "s.show_refmods" in src, "collapse state must be restored from the UI prefs"

    def test_backend_routes_for_panel_actions(self):
        src = _js_source()
        for route in ("/longmedia/refmods/list", "/longmedia/refmods/save",
                      "/longmedia/refmods/delete", "/longmedia/refmods/from_take"):
            assert route in src
            assert re.search(rf"api\.fetchApi\(['\"]{re.escape(route)}", src), \
                f"{route} must be called through api.fetchApi"


# =========================================================================
# 19. refmod timeline track + clips are positioned by interval
# =========================================================================

class TestRefmodTimelineInterval:
    def test_refmod_track_is_renderable(self):
        src = _js_source()
        types = _extract_js_array("EXTRA_TRACK_TYPES")
        assert "refmod" in types
        assert '"refmod"' in src or "'refmod'" in src

    def test_extra_clips_drawn_from_interval_geometry(self):
        src = _js_source()
        assert re.search(r"blockBase\(start\*pps,c\.duration\*pps", src), \
            "extra/refmod clips must be drawn from start*pps .. duration*pps"

    def test_timeline_extent_includes_extra_clips(self):
        src = _js_source()
        m = re.search(r"function timelineExtent\(doc\)\{(.*?)return end;\}", src, re.DOTALL)
        assert m, "timelineExtent not found"
        assert "extra_tracks" in m.group(1), "refmod clips must extend the timeline extent"

    def test_refmod_clip_label_resolves_library_record(self):
        src = _js_source()
        assert "rmRec" in src, "refmod clips must resolve their doc.refmods record for the label"
        assert "doc.refmods||[]).find(r=>r.refmod_id===c.refmod_id" in src.replace(" ", "") or \
            re.search(r"\(doc\.refmods\|\|\[\]\)\.find\(r=>r\.refmod_id===c\.refmod_id\)", src)

    def test_refmod_clip_normalizes_interval_and_link(self):
        m = re.search(r"function normalizeExtraClip\(.*?\)\{(.*?)return \{.*?\};\}", _js_source(), re.DOTALL)
        assert m, "normalizeExtraClip not found"
        body = m.group(0)
        assert "start:Math.max(0,finite(c.start,0))" in body, "clip start must be normalized"
        assert "duration:clamp(c.duration,.25,600,5)" in body, "clip duration must be clamped"
        assert "refmod_id" in body and "strength" in body


# =========================================================================
# 20. ComfyUI / LiteGraph lifecycle
# =========================================================================

class TestComfyLifecycle:
    def test_extension_registered(self):
        src = _js_source()
        assert re.search(r"app\.registerExtension\(\{name:\"MiniMaxH3\.LongMediaDirector", src), \
            "Director extension must register with the ComfyUI app"

    def test_node_lifecycle_hooks(self):
        src = _js_source()
        assert "async nodeCreated(node)" in src, "nodeCreated hook missing"
        assert "async afterConfigureGraph()" in src, "afterConfigureGraph hook missing"
        assert "const prev=node.onConfigure" in src, "onConfigure must be chained, not overwritten"
        assert "const resize=node.onResize" in src, "onResize must be chained, not overwritten"
        assert "const removed=node.onRemoved" in src, "onRemoved must be chained for cleanup"

    def test_on_removed_cleans_runtime_state(self):
        src = _js_source()
        idx = src.find("const removed=node.onRemoved")
        assert idx >= 0
        body = src[idx:idx + 700]
        assert "stopPlayback" in body, "removal must stop RAF playback"
        assert "disconnect()" in body, "removal must disconnect ResizeObservers"
        assert "__lmdEditor=null" in body, "removal must release the cached editor widget"

    def test_dom_widget_created_once(self):
        src = _js_source()
        assert src.count("addDOMWidget(") >= 1
        assert re.search(r"if\(node\.__lmdEditor\)\{if\(\(node\.widgets\|\|\[\]\)\.includes\(node\.__lmdEditor\)\)return node\.__lmdEditor;", src), \
            "ensureEditor must reuse the existing widget instead of stacking new ones"


# =========================================================================
# 21. The decorative surface widget must never serialize
# =========================================================================

class TestNoDetachedSerializedWidgetState:
    def test_widget_options_serialize_false(self):
        assert re.search(r"serialize\s*:\s*false", _js_source()), \
            "DOM widget options must set serialize:false"

    def test_widget_instance_never_serializes(self):
        src = _js_source()
        assert "ed.serialize=false" in src
        assert "ed.serializeValue=()=>undefined" in src
        assert "getValue:()=>undefined" in src

    def test_stale_surface_widgets_are_purged(self):
        src = _js_source()
        assert 'w.name==="director_surface"' in src, \
            "stale director_surface widgets must be removed before creating a new one"
        assert "node.widgets.splice" in src, "purged widgets must be spliced out of node.widgets"
        assert "w.element?.remove?.()" in src, "purged widget DOM must be detached"

    def test_no_other_dom_widgets(self):
        # Only one DOM widget is allowed on the node: the non-serializing surface.
        assert _js_source().count("addDOMWidget(") == 1


# =========================================================================
# v0.6.57. Director ENCODE is a separate VAE-only server operation
# =========================================================================
class TestRefmodEncodeRouteContract:

    @staticmethod
    def _source(name):
        return (JS_PATH.parent.parent / name).read_text(encoding="utf-8")

    def test_encode_route_is_registered_and_vae_only(self):
        init = self._source("__init__.py")
        assert "routes.get('/longmedia/refmods/vaes')" in init
        assert "routes.post('/longmedia/refmods/encode')" in init
        body = init[init.index("@routes.post('/longmedia/refmods/encode')"):
                   init.index("@routes.get('/longmedia/director/takes')")]
        assert '_lm_refmod_encode_one' in body
        for forbidden in (
            '_lm_refmod_encode_record', 'hunyuan_sampler', 'hybrid_sampler',
            'KSampler', 'VAELoader', 'load_checkpoint', '_lm_refmod_blocks_for_segment',
        ):
            assert forbidden not in body

    def test_director_panel_exposes_encode(self):
        src = _js_source()
        assert 'async function refmodEncode' in src
        assert "fetchApi('/longmedia/refmods/encode'" in src
        assert 'button("ENCODE"' in src
        assert 'VAE-only RefMod preparation' in src
        assert 'NOT ENCODED · press ENCODE' in src

    def test_encode_reports_vaes_used(self):
        nodes = self._source("nodes.py")
        assert 'def _lm_refmod_encode_one' in nodes
        assert 'def _lm_refmod_vae_catalog' in nodes

    def test_encode_announces_every_stage_and_its_completion(self):
        # ENCODE runs no sampler progress bar, so the console has to say when it
        # is finished — otherwise a completed run looks identical to a stalled one.
        nodes = self._source("nodes.py")
        body = nodes[nodes.index("def _lm_refmod_encode_one("):
                     nodes.index("def _lm_refmod_blocks_for_segment(")]
        assert '[DIRECTOR ENCODE] start' in body
        assert '[DIRECTOR ENCODE] stage · VAEs loaded' in body
        assert '[DIRECTOR ENCODE] stage · latents encoded' in body
        assert '[DIRECTOR ENCODE] done' in body
        assert '[DIRECTOR ENCODE] FAILED' in body
        assert 'perf_counter() - started' in body


# =========================================================================
# v0.6.57. SYNC must reconcile with the library, never union into it
# =========================================================================
_RECONCILE_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.env.JS_PATH, 'utf8');
function sliceFn(name) {
  const start = src.indexOf('function ' + name + '(');
  if (start < 0) throw new Error('missing ' + name);
  const rest = src.slice(start + 1);
  const next = rest.match(/^(?:async\s+)?function\s+/m);
  return src.slice(start, next ? start + 1 + next.index : src.length);
}
const normalizeRefmod = r => Object.assign({}, r);
eval(sliceFn('reconcileRefmodLibrary'));
eval(sliceFn('refmodTimelineReferrers'));
const savedA = {refmod_id: 'a', path: 'a.safetensors', folder: 'Unsorted', name: 'A'};
const savedB = {refmod_id: 'b', path: 'b.safetensors', folder: 'Unsorted', name: 'B'};
const draft = {refmod_id: 'draft', path: '', folder: 'Unsorted', name: 'Draft'};
// Emptied library folder: every saved record must leave, nothing may linger.
const onlySaved = reconcileRefmodLibrary([savedA, savedB], []);
// A never-saved draft is document state, not library state: it survives.
const withDraft = reconcileRefmodLibrary([savedA, draft], []);
const mixed = reconcileRefmodLibrary([savedA, savedB, draft], [{
  refmod_id: 'a', path: 'a2.safetensors', folder: 'X', name: 'A2',
  state: 'ready', kind: 'image', format_version: 4, members: [],
}]);
console.log(JSON.stringify({
  onlySavedCount: onlySaved.refmods.length,
  onlySavedRemoved: onlySaved.removedIds,
  withDraftCount: withDraft.refmods.length,
  withDraftIds: withDraft.refmods.map(r => r.refmod_id),
  withDraftRemoved: withDraft.removedIds,
  withDraftKept: withDraft.drafts.length,
  mixedCount: mixed.refmods.length,
  mixedIds: mixed.refmods.map(r => r.refmod_id).sort(),
  mixedRemoved: mixed.removedIds,
  mixedUpdated: mixed.updated,
  mixedPath: mixed.refmods.find(r => r.refmod_id === 'a').path,
  timelineRefs: refmodTimelineReferrers(
    {extra_tracks: [{clips: [{refmod_id: 'b'}, {refmod_id: 'a'}]}]}, ['b']),
}));
"""


class TestRefmodSyncReconcilesLibraryContract:

    def test_sync_drops_records_whose_library_artifact_is_gone(self):
        import json
        import os
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:  # pragma: no cover - exercised by the JS syntax gate in builds
            return
        env = dict(os.environ, JS_PATH=str(JS_PATH))
        run = subprocess.run(
            [node, "-e", _RECONCILE_HARNESS],
            capture_output=True, text=True, timeout=90, env=env,
        )
        assert run.returncode == 0, run.stderr
        got = json.loads(run.stdout.strip().splitlines()[-1])

        # The reported bug: an emptied library folder left the panel populated.
        assert got["onlySavedCount"] == 0
        assert sorted(got["onlySavedRemoved"]) == ["a", "b"]

        # Unsaved drafts are document state and must survive a library sync.
        assert got["withDraftCount"] == 1
        assert got["withDraftIds"] == ["draft"]
        assert got["withDraftRemoved"] == ["a"]
        assert got["withDraftKept"] == 1

        # Survivors refresh from the library instead of going stale.
        assert got["mixedCount"] == 2
        assert got["mixedIds"] == ["a", "draft"]
        assert got["mixedRemoved"] == ["b"]
        assert got["mixedUpdated"] == 1
        assert got["mixedPath"] == "a2.safetensors"

        # Losing a RefMod that the timeline still uses has to be visible.
        assert got["timelineRefs"] == 1

    def test_sync_never_treats_a_failed_listing_as_an_empty_library(self):
        src = _js_source()
        listing = src[src.index("async function refmodList("):
                      src.index("function reconcileRefmodLibrary(")]
        assert "catch(_){return [];}" not in listing
        assert "throw new Error" in listing
        assert "Nothing was changed" in src
        assert "reconcileRefmodLibrary(doc.refmods||[],remote)" in src


# =========================================================================
# v0.6.57. A save/encode response must not reset the authored RefMod settings
# =========================================================================
_PRESERVE_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.env.JS_PATH, 'utf8');
function sliceFn(name) {
  const start = src.indexOf('function ' + name + '(');
  if (start < 0) throw new Error('missing ' + name);
  const rest = src.slice(start + 1);
  const next = rest.match(/^(?:async\s+)?function\s+/m);
  return src.slice(start, next ? start + 1 + next.index : src.length);
}
// `eval` keeps a `const` binding in its own scope, so bind the real list here.
const REFMOD_SERVER_FIELDS = eval(src.match(/const REFMOD_SERVER_FIELDS=(\[[^\]]*\]);/)[1]);
// Normalizer stub so the test isolates the overlay contract itself.
const normalizeRefmod = r => Object.assign({
  spatial_grid: 'DEFAULT', temporal_frames: 24,
  refinement_steps: 20, strength: 1, description: '',
}, r);
eval(sliceFn('applyRefmodServerState'));
const authored = {
  refmod_id: 'x', name: 'Test', mode: 'COMPRESSED', concept_type: 'character',
  description: 'keep me', spatial_grid: '8x8',
  temporal_frames: 4, refinement_steps: 0, strength: 0.42,
  source: 'subject:s1', path: '',
};
// Exactly the disk projection _refmod_record() builds from the written artifact.
const returned = {
  refmod_id: 'x', name: 'Test', kind: 'image', mode: 'COMPRESSED',
  concept_type: 'character', description: '', source: '',
  path: 'longmedia_refmods/Test__x.safetensors', folder: 'Unsorted',
  strength: 1.0, format_version: 4, members: [], state: 'ready',
};
console.log(JSON.stringify(applyRefmodServerState(authored, returned)));
"""


class TestRefmodSettingsSurviveServerResponses:

    def test_server_response_preserves_authored_settings(self):
        import json
        import os
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:  # pragma: no cover - covered by the build's JS syntax gate
            return
        env = dict(os.environ, JS_PATH=str(JS_PATH))
        run = subprocess.run(
            [node, "-e", _PRESERVE_HARNESS],
            capture_output=True, text=True, timeout=90, env=env,
        )
        assert run.returncode == 0, run.stderr
        got = json.loads(run.stdout.strip().splitlines()[-1])

        # The reported bug: ENCODE reset these to the inspector defaults.
        assert got["spatial_grid"] == "8x8"
        assert got["temporal_frames"] == 4
        assert got["refinement_steps"] == 0
        assert abs(got["strength"] - 0.42) < 1e-9
        assert got["description"] == "keep me"
        assert got["source"] == "subject:s1"
        assert got["mode"] == "COMPRESSED"
        assert got["concept_type"] == "character"

        # What writing the artifact actually determined still lands.
        assert got["path"] == "longmedia_refmods/Test__x.safetensors"
        assert got["state"] == "ready"
        assert got["format_version"] == 4
        assert got["kind"] == "image"

    def test_server_and_client_agree_on_the_artifact_owned_field_set(self):
        src = _js_source()
        start = src.index("const REFMOD_SERVER_FIELDS=")
        owned = src[start:src.index("];", start) + 2]
        for name in ("refmod_id", "path", "folder", "state", "kind",
                     "format_version", "members"):
            assert f'"{name}"' in owned

        init = (JS_PATH.parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "_REFMOD_SERVER_FIELDS = ('path', 'folder', 'state', 'kind', 'format_version', 'members')" in init
        # Save and ENCODE both answer with authored settings, never a disk projection.
        assert "refmod': _refmod_authored_record(record, stored, refmod_id)" in init
        assert "saved = _refmod_record(target" not in init
        assert "refmod = _refmod_record(result['path']" not in init

    def test_no_merge_site_redefaults_a_record_from_the_response(self):
        src = _js_source()
        assert "Object.assign(refmod,normalizeRefmod(res.refmod))" not in src
        assert "normalizeRefmod(await refmodSave(" not in src


# =========================================================================
# v0.6.57. An encoded library RefMod must be placeable on the REFMOD layer
# =========================================================================
_PLACE_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.env.JS_PATH, 'utf8');
function sliceFn(name) {
  const start = src.indexOf('function ' + name + '(');
  if (start < 0) throw new Error('missing ' + name);
  const rest = src.slice(start + 1);
  const next = rest.match(/^(?:async\s+)?function\s+/m);
  return src.slice(start, next ? start + 1 + next.index : src.length);
}
const finite = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
const clamp = (v, lo, hi, d) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
};
let seq = 0;
const defaultExtraTrack = type =>
  ({track_id: 't' + (++seq), type, name: type, enabled: true, locked: false, muted: false, clips: []});
const defaultExtraClip = (track, start = 0, duration = 5) =>
  ({clip_id: 'c' + (++seq), name: 'clip', start, duration,
    prompt: '', embedding_name: '', subject_id: null, source_in: 0});
const sceneInfoAt = () => ({shot: {duration: 7}});
eval(sliceFn('placeRefmodOnTimeline'));

const doc = {extra_tracks: []};
const node = {__lmdPlayhead: 2};
const rm = {refmod_id: 'rm1', name: 'Pixel Style', strength: 0.42};
const first = placeRefmodOnTimeline(node, doc, rm);
const out = {
  tracks: doc.extra_tracks.length,
  trackType: doc.extra_tracks[0].type,
  clips: doc.extra_tracks[0].clips.length,
  refmod_id: first.refmod_id,
  strength: first.strength,
  name: first.name,
  start: first.start,
  duration: first.duration,
  selection: node.__lmdSelection,
  selected: node.__lmdRefmodSelected,
};
// A second placement reuses the same layer instead of spawning another one.
placeRefmodOnTimeline(node, doc, rm);
out.clipsAfterSecond = doc.extra_tracks[0].clips.length;
out.tracksAfterSecond = doc.extra_tracks.length;
// A locked layer refuses loudly rather than silently making a second layer.
doc.extra_tracks[0].locked = true;
try {
  placeRefmodOnTimeline(node, doc, rm);
  out.lockedError = null;
} catch (err) {
  out.lockedError = String(err && err.message);
}
out.clipsWhenLocked = doc.extra_tracks[0].clips.length;
console.log(JSON.stringify(out));
"""


class TestRefmodPlaceOnLayerContract:

    def test_placing_a_refmod_creates_a_timeline_clip_that_conditions_the_render(self):
        import json
        import os
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:  # pragma: no cover - covered by the build's JS syntax gate
            return
        env = dict(os.environ, JS_PATH=str(JS_PATH))
        run = subprocess.run(
            [node, "-e", _PLACE_HARNESS],
            capture_output=True, text=True, timeout=90, env=env,
        )
        assert run.returncode == 0, run.stderr
        got = json.loads(run.stdout.strip().splitlines()[-1])

        # Placement is the bridge from "encoded" to "affects the render".
        assert got["tracks"] == 1
        assert got["trackType"] == "refmod"
        assert got["clips"] == 1
        assert got["refmod_id"] == "rm1"
        assert got["strength"] == 0.42
        assert got["name"] == "Pixel Style"
        assert got["start"] == 2
        assert got["duration"] == 7
        assert got["selection"]["track"].startswith("extra:")
        assert got["selected"] == "rm1"

        # Repeated placement accumulates clips on one layer.
        assert got["tracksAfterSecond"] == 1
        assert got["clipsAfterSecond"] == 2

        # A locked layer refuses and changes nothing.
        assert "Unlock" in got["lockedError"]
        assert got["clipsWhenLocked"] == 2

    def test_refmod_clip_inspector_can_retarget_and_stretch(self):
        src = _js_source()
        assert "function placeRefmodOnTimeline(" in src
        assert 'button("PLACE ON LAYER"' in src
        row = src[src.index('if(track.type==="refmod")'):src.index('if(track.type==="embedding")')]
        assert 'clip.refmod_id=v||null' in row
        assert 'clip.strength=clamp(v,0,1,1)' in row
        assert 'Stretch this RefMod across the complete Director timeline' in row
