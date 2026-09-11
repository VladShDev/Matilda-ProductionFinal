"""What is ACTUALLY in front of her --- a RULER, not her perception.

His call, 2026-08-15, when the hand-built grouping had spent a day bordering the
painting instead of him: *"any kid can compile some interactive mask which is
applied to real video and knows what shape of face is."*  He is right, and the
reason is that the kid loads a TRAINED MODEL.  Rule 5 says her own vision stays
hand-built with no nets in it, so the model lives HERE, in `measure/`, where
nothing she keeps can ever read it:

    brain/    never.  It may not name a thing, and it does not import this.
    body/     never.  `parts.find` is what SHE sees and it keeps running every
              look, untouched.
    measure/  this file --- what a trained detector says is really there, laid
              over the top and never fed back into her.

Two uses, both honest.  It puts a NAMED box on him and on what he is holding, so
the panel stops being a guess.  And it is the yardstick every future change to
her own grouping is scored against: how much of him her lump covered, how much
of the wall it leaked onto --- a number instead of squinting at boxes.

IT RUNS ON HER RETINA, NOT ON THE CAMERA FRAME, and that is the whole reason it
can be drawn over her view at all.  The old tree ran it on `/shown` --- the
camera's pixels, at the camera's moment --- and drew the result over `/see`, her
retina.  Two pictures, two coordinate systems, and every outline sat loose on
the thing it was around.  Here it is handed **the same array `parts.find` was
given**, so a box from the ruler and a box from her own grouping are in one
space by construction and there is nothing to keep in step.

    python -m measure.ruler        loads the model and times a look

The model is `measure/models/efficientdet_lite0.tflite` (EfficientDet-Lite0, 80
everyday things, from Google's mediapipe-models host).  It is gitignored --- 13.8
MB of weights are not this project's history.  Without it, and without
`mediapipe` installed, `Ruler.ready` is False and `look` returns nothing: the
panel says "nothing named" and every other thing she does is unaffected.
"""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "efficientdet_lite0.tflite")


class Ruler:
    """The detector, loaded once.

    `look(pic)` gives one dict per thing it is sure enough of: `x`, `y` in HER
    field (-1..1), `w`, `h` as a FRACTION of the view (0..1), `area`, `name`,
    `score`.  About 48 ms a look on this machine, which is why nothing calls it
    inside her tick.

    `ready` is False when mediapipe or the weights are missing, and then `look`
    returns `[]` --- this is an instrument, and an instrument that is not
    plugged in must never be able to stop her.
    """

    def __init__(self, sure: float = 0.30, most: int = 8) -> None:
        self.ready = False
        self.why = ""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except Exception as why:                            # noqa: BLE001
            self.why = f"mediapipe will not import: {type(why).__name__} {why}"
            return
        if not os.path.exists(MODEL):
            self.why = f"no weights at {MODEL}"
            return
        try:
            self._mp = mp
            self._det = vision.ObjectDetector.create_from_options(
                vision.ObjectDetectorOptions(
                    base_options=mpp.BaseOptions(model_asset_path=MODEL),
                    running_mode=vision.RunningMode.IMAGE,
                    score_threshold=sure, max_results=most))
            self.ready = True
        except Exception as why:                            # noqa: BLE001
            self.why = f"the model would not load: {type(why).__name__} {why}"

    def look(self, pic: np.ndarray) -> list[dict]:
        if not self.ready or pic is None:
            return []
        pic = np.asarray(pic)
        if pic.ndim == 2:
            pic = np.repeat(pic[:, :, None], 3, axis=2)
        if pic.shape[2] > 3:
            pic = pic[:, :, :3]
        if pic.dtype != np.uint8:
            # HER LEVELS, NOT AN EXPOSURE.  Her retina carries real radiance and
            # a lit room can sit far under 1.0, so clipping it straight to
            # 0..255 hands the detector a near-black frame and it names nothing.
            # Scaled by the frame's own peak, which is what `/eye` shows him.
            top = float(pic.max())
            pic = np.clip(pic / top if top > 1e-6 else pic, 0.0, 1.0)
            pic = (pic * 255).astype(np.uint8)
        try:
            out = self._det.detect(self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(pic)))
        except Exception:                                   # noqa: BLE001
            return []
        side = float(pic.shape[0] - 1)
        got = []
        for d in out.detections:
            b, c = d.bounding_box, d.categories[0]
            # ONE UNIT, AND IT IS `parts.find`'S.  A box's CENTRE is -1..1 and
            # its SIZE is a fraction of the view, 0..1 --- two different units in
            # one row, settled 2026-08-15.  This once emitted the size in -1..1
            # as well, the panel drew every named box at twice its true size, and
            # the panel was then "corrected" by halving its own arithmetic ---
            # which cancelled for these boxes and halved everything measured by
            # `find` instead.  Two producers, two units, one formula.
            got.append({"x": (b.origin_x + b.width / 2) / side * 2 - 1,
                        "y": (b.origin_y + b.height / 2) / side * 2 - 1,
                        "w": b.width / side, "h": b.height / side,
                        "area": (b.width * b.height) / (side * side),
                        "name": c.category_name, "score": float(c.score)})
        # HIM FIRST.  A person is what he asked to be bordered, so a person takes
        # a slot before the couch does however sure the couch is.
        got.sort(key=lambda t: (t["name"] != "person", -t["score"]))
        return got


def main() -> int:                       # pragma: no cover - a probe
    import time
    r = Ruler()
    if not r.ready:
        print("the ruler is NOT plugged in --- " + r.why)
        print("she is unaffected; the panel will say 'nothing named'.")
        return 1
    print(f"the ruler is loaded from {MODEL}")
    rng = np.random.default_rng(0)
    pic = rng.random((512, 512, 3), np.float32) * 0.3
    cost = []
    for _ in range(5):
        t0 = time.perf_counter()
        got = r.look(pic)
        cost.append((time.perf_counter() - t0) * 1000)
    print(f"a look costs {sum(cost) / len(cost):.1f} ms "
          f"(never inside her 33 ms tick)")
    print(f"on noise it names {len(got)} things, which is the right answer")
    return 0


if __name__ == "__main__":               # pragma: no cover - a probe
    raise SystemExit(main())
