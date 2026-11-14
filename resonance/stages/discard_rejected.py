"""discard_rejected: ClipManifest -> ClipManifest. Deletes the audio of every clip the batch's filters rejected.

About half of a batch's segmented clips fail the filters, and their wavs would otherwise stay on disk for good.
The manifest passes through unchanged. After this stage the filters cannot rerun on the batch without segmenting
it again, since the rejected audio is gone.
"""

from resonance.io import manifest
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage


class DiscardRejected(Stage):
    name = "discard_rejected"
    scope = "batch"
    consumes = ClipManifest
    produces = ClipManifest
    needs = ("segment.csv",)

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "discard_rejected.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        kept = manifest.read(source.path)
        names = {row["name"] for row in kept}
        deleted, freed = 0, 0
        for row in manifest.read(ctx.work_dir / "segment.csv"):
            path = ctx.layout.root / row["audio_path"]
            if row["name"] not in names and path.exists():
                freed += path.stat().st_size
                path.unlink()
                deleted += 1
        manifest.write(ctx.work_dir / "discard_rejected.csv", kept)
        ctx.summary.update(clips_out=len(kept), hours_out=round(manifest.hours(kept), 3), deleted=deleted, freed_gb=round(freed / 1024 ** 3, 2))
        return self.output(ctx)
