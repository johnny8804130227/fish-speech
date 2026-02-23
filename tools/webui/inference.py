import html
import time
from functools import partial
from typing import Any, Callable

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


def inference_wrapper(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    When chunk_length > 0, yields intermediate audio segments for real-time playback.
    """

    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    use_streaming = chunk_length > 0

    req = ServeTTSRequest(
        text=text,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
        streaming=use_streaming,
    )

    t0 = time.monotonic()
    t_first = None
    last_t = None
    intervals = []
    segment_count = 0

    for result in engine.inference(req):
        match result.code:
            case "header":
                pass
            case "segment":
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                if last_t is not None:
                    intervals.append(now - last_t)
                last_t = now
                segment_count += 1

                stats = _format_stats(t0, t_first, intervals, segment_count, done=False)
                yield result.audio, None, stats
            case "final":
                stats = _format_stats(t0, t_first, intervals, segment_count, done=True)
                if segment_count == 0:
                    # non-streaming path: engine collected everything into "final"
                    yield result.audio, None, stats
                else:
                    yield None, None, stats
                return
            case "error":
                stats = _format_stats(t0, t_first, intervals, segment_count, done=True)
                yield None, build_html_error_message(i18n(result.error)), stats
                return

    stats = _format_stats(t0, t_first, intervals, segment_count, done=True)
    yield None, i18n("No audio generated"), stats


def _format_stats(
    t0: float,
    t_first: float | None,
    intervals: list[float],
    segment_count: int,
    done: bool,
) -> str:
    elapsed = time.monotonic() - t0
    first = "n/a" if t_first is None else f"{t_first - t0:.2f}s"
    avg = (
        f"{sum(intervals) / len(intervals):.2f}s" if intervals else "n/a"
    )
    status = "done" if done else "streaming"
    return (
        f"Status: {status}\n"
        f"Segments: {segment_count}\n"
        f"First segment: {first}\n"
        f"Avg segment interval: {avg}\n"
        f"Elapsed: {elapsed:.2f}s"
    )


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red;
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )
