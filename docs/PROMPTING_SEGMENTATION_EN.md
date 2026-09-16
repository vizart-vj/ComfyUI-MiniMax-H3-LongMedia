# Fixed Segmentation Prompting Rules

Use fixed segmentation when:

```text
timeline_mode = segmented
```

Segmentation is an internal execution strategy for one continuous semantic movie. It is not a storyboard scheduler.

## Mental Model

LongMedia creates fixed-duration internal units from `segment_duration` and carries H3 continuation context across them.

The prompt should describe the **final continuous movie**, not the hidden segmentation.

Good:

```text
The woman walks steadily through the corridor while the surrounding lights gradually become warmer.
Her pace and direction remain consistent as the environment develops around her.
```

Avoid writing instructions such as "segment 1 starts" or "each segment resets". The model should not be asked to reenact LongMedia's internal bookkeeping.

## Segment Duration

`segment_duration` is the amount of new visible timeline produced by each fixed unit.

`transition_frames` is hidden continuation context and does not subtract from that visible duration.

Shorter segments usually:

- reduce peak sequence geometry;
- improve control on constrained GPUs;
- create more continuation boundaries.

Longer segments usually:

- reduce boundary count;
- increase packed sequence/workspace size.

## Continuous Actions

Use temporal continuation language:

```text
continues walking
keeps singing
gradually turns
maintains the same direction
the illumination develops steadily
```

Describe starts only when an action really starts at that point in the final movie.

## References

The conditioning family still comes from `h3_mode`.

Examples:

- `hybrid + segmented` — opening-keyframe-driven continuous movie;
- `ref2va + segmented` — reference-driven continuous movie;
- manual mode — advanced custom diagnostics.

`video_ref_edit` normally uses a single source timeline; use the Video Reconstructor contract for source-video reconstruction/edit plans that require segmented source windows.

## Audio

When `audio_mode=lip_sync`, Audio1 owns the performance timing. LongMedia slices/aligns the active source timeline internally; do not manually split phonemes at segment boundaries.

For preserve-style modes, remember that a Video IMAGE input does not contain soundtrack data. Connect source audio separately.

## When to Use MultiClip Instead

Choose MultiClip when individual sections need their own:

- prompt;
- duration;
- seed;
- camera card;
- explicit storyboard identity.

Choose Segmented when the movie is fundamentally one continuous prompt and segmentation is primarily an execution/VRAM decision.
