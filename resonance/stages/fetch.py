"""fetch: batch reference -> RawBatch. Downloads and unpacks one batch of the corpus."""

from resonance.datasets import yodas2_sidon
from resonance.pipeline.artifacts import BatchRef, RawBatch
from resonance.pipeline.stage import Context, Stage


class Fetch(Stage):
    name = "fetch"
    scope = "batch"
    consumes = BatchRef
    produces = RawBatch

    def output(self, ctx: Context) -> RawBatch:
        return RawBatch(ctx.layout.raw(ctx.batch))

    def run(self, source: BatchRef, ctx: Context) -> RawBatch:
        from huggingface_hub import get_token

        target = ctx.layout.raw(source)
        count = len(yodas2_sidon.recordings(target)) if target.is_dir() else 0
        if not count:       # an unpacked batch already on disk is used as it is
            archive = target.with_name(f"{target.name}.tar.gz")
            archive.parent.mkdir(parents=True, exist_ok=True)
            yodas2_sidon.download(f"{source.language}{source.shard}", source.name, archive, get_token())
            count = yodas2_sidon.extract(archive, target)
            archive.unlink()
        if not count:
            raise RuntimeError(f"{source.key} holds no recordings")
        ctx.summary["recordings"] = count
        return RawBatch(target)
