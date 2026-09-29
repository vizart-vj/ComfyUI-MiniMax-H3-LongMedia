# ComfyUI-MiniMax-H3-LongMedia 0.6.63

## Director: shared Camera presets

- Create a preset from any CAMERA clip, apply it to another clip, rename or update it with `OVERWRITE`, and remove it with `DELETE`.
- The library is stored in `director_json`, survives workflow reloads, and is available to existing and newly created CAMERA clips.
- The inspector reports when a clip's settings have changed since applying its preset.

## Regeneration and conditioning

- Changing an AUDIO clip role from `diegetic` to `reactive` is supported. It changes generation guidance and can invalidate that clip's TAKE.
- Conditioning copies now turn absent optional branches into empty lists before guider preprocessing. A missing required `positive` branch raises a clear error instead of the `'NoneType' object is not iterable` traceback.

Existing Director workflows remain compatible; a missing camera preset library is initialized as empty.
