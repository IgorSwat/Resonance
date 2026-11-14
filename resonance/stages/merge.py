"""merge: list of batch ClipManifests -> one ClipManifest. Concatenates the batches of a run into one pool."""

from resonance.io import manifest
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage


class Merge(Stage):
    name = "merge"
    scope = "pool"
    consumes = list[ClipManifest]
    produces = ClipManifest

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "merge.csv")

    def run(self, source: list[ClipManifest], ctx: Context) -> ClipManifest:
        missing = [str(part.path) for part in source if not part.path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} batch manifests are missing, first: {missing[0]}")
        rows = [row for part in source for row in manifest.read(part.path)]
        duplicates = len(rows) - len({row["name"] for row in rows})
        if duplicates:
            raise ValueError(f"{duplicates} clip names occur in more than one batch")
        manifest.write(ctx.work_dir / "merge.csv", rows)
        ctx.summary.update(batches=len(source), clips_out=len(rows), hours_out=round(manifest.hours(rows), 3),
                           speakers=len({row["speaker_id"] for row in rows}))
        return self.output(ctx)
