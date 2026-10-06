"""The fine-tune's loader falls back to no worker processes when its workers crash, and
carries on from the same step."""
import pytest

from mycomap_vision.finetune import batches_with_fallback, is_worker_crash

CRASH = RuntimeError("DataLoader worker (pid(s) 31468) exited unexpectedly")


class Loaders:
    """make_loader stand-in: each call returns the next scripted loader and is recorded."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls = []

    def __call__(self, workers, steps_left, seed):
        self.calls.append((workers, steps_left, seed))
        script = self.scripts.pop(0)

        def loader():
            for item in script:
                if isinstance(item, BaseException):
                    raise item
                yield item
        return loader()


def run(loaders, workers=6, steps=5, seed=7):
    state = {"step": 0, "fallbacks": []}
    got = []
    for batch in batches_with_fallback(loaders, workers, steps, seed, lambda: state["step"],
                                       lambda at, err: state["fallbacks"].append(at)):
        got.append(batch)
        state["step"] += 1
        if state["step"] >= steps:
            break
    return got, state["fallbacks"]


def test_a_worker_crash_carries_on_from_the_same_step_with_no_workers():
    loaders = Loaders(["b1", "b2", CRASH], ["b3", "b4", "b5"])
    got, fallbacks = run(loaders)
    assert got == ["b1", "b2", "b3", "b4", "b5"]
    assert fallbacks == [2]
    assert loaders.calls == [(6, 5, 7), (0, 3, 9)], "only the steps left, reseeded by the step"


def test_a_crash_before_the_first_batch_falls_back_too():
    loaders = Loaders([CRASH], ["b1", "b2"])
    got, fallbacks = run(loaders, steps=2)
    assert got == ["b1", "b2"]
    assert fallbacks == [0]


def test_an_error_raised_inside_a_worker_is_a_crash_too():
    caught = PermissionError("Caught PermissionError in DataLoader worker process 0.")
    assert is_worker_crash(caught)
    got, fallbacks = run(Loaders([caught], ["b1"]), steps=1)
    assert got == ["b1"] and fallbacks == [0]


def test_with_no_workers_a_crash_is_a_real_error_and_is_raised():
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        run(Loaders([CRASH]), workers=0)


def test_a_second_crash_after_falling_back_is_raised():
    with pytest.raises(RuntimeError):
        run(Loaders(["b1", CRASH], [CRASH]))


def test_other_errors_are_not_mistaken_for_a_worker_crash():
    assert not is_worker_crash(ValueError("bad label"))
    assert not is_worker_crash(RuntimeError("CUDA out of memory"))
    with pytest.raises(KeyError):
        run(Loaders(["b1", KeyError("species")]))


def test_without_a_crash_the_workers_are_kept():
    loaders = Loaders(["b1", "b2", "b3"])
    got, fallbacks = run(loaders, steps=3)
    assert got == ["b1", "b2", "b3"] and fallbacks == []
    assert loaders.calls == [(6, 3, 7)]
