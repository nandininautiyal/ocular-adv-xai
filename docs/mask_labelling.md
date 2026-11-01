# Hand-labelling eye masks

This guide is for the team members who will hand-label VISOB eye images, so that we
can check the learned eye masks (`scripts/17_make_eye_masks.py`) before using them for
region-level results.

## Why we label

The learned masks come from a model trained on face photos, not on VISOB, and nobody
has checked them on our images yet. Before we write "attention moves from the iris to
the skin" in a paper, we must show that the masks agree with careful human labels,
and how much better they are than the old heuristic masks (`src/xai/masks.py`).
About 60 labelled images are enough for that, with confidence intervals.

## The order of work

1. **Kaggle:** make the masks with `17_make_eye_masks.py` (see its header).
2. **Kaggle:** make the labelling pack with
   `python scripts/18_validate_masks.py --export-sample --csv <image list> --masks <masks_<tag>.npz> --out-dir /kaggle/working/label_pack --zip`,
   then download `label_pack.zip`.
3. **Laptop:** label the images (this guide).
4. **Laptop:** run the validation (last section).

## Data rules

- These are biometric images, and the licence of our Kaggle copy of VISOB is unclear.
  The team must agree that copying 60 images to a laptop is acceptable before step 2.
- Keep the pack on your own laptop, outside OneDrive, Google Drive, Dropbox or any
  other synced folder. Share it inside the team only, for example on a USB stick.
- Use **local tools only**. Never upload the images to a hosted labelling service:
  no app.cvat.ai, no Label Studio cloud or HumanSignal, no Roboflow, Supervisely or
  similar websites.
- Delete the images when the project ends.

## What is in the pack

| Path | What it is |
|---|---|
| `images/<name>` | The image to label: the stored 240 x 160 crop, unchanged. |
| `prefill/<name>` | The model's mask as a colour picture. Start from it, but do not trust it. |
| `overlays/<name>` | The image with the prefill on top, to look at. |
| `labels/` | Empty. Save your finished labels here. |
| `sample.csv` | One row per image: name, relpath, subject, phone, lighting, eye, and `prefilled`. |
| `cvat_prefill.zip` | The prefill in CVAT's "Segmentation mask 1.1" format. |
| `labelmap.txt` | The label colours in CVAT's format. |
| `model_masks_sample.npz` | For the validation script only. Do not open it. |

`<name>` is the image's relpath with `/` replaced by `__`, for example
`oppo_day_light_short__enrollment_right_eye__1141__1141_1005_r_c.png`. Your label
must have exactly the same name.

About one image in five has `prefilled = False` in `sample.csv`. These "blind" images
have no prefill and no overlay. Label them from scratch, and do not look at the model's
masks for them. Comparing blind and prefilled images tells us how much starting from
the model's answer pulls the labels towards it.

## The classes

Every pixel gets exactly one class. Use these exact colours, with no anti-aliasing
(no soft or blended edges).

| Id | Class | Colour (RGB, hex) | What it covers |
|---|---|---|---|
| 0 | skin | black (0, 0, 0), `#000000` | Everything that is not one of the classes below: cheek and nose skin, the **lower** lid and its lashes, hair, glasses frames. You never paint skin on purpose: it is what is left. |
| 1 | iris | blue (0, 0, 255), `#0000FF` | The coloured ring **and the pupil**, with any light reflections inside them. Only the visible part: where the lid or a solid band of lashes covers the iris, that part is lid. |
| 2 | sclera | green (0, 255, 0), `#00FF00` | The visible white of the eye, red vessels included, up to the lid margins. Not the pink corner tissue (that is caruncle). |
| 3 | lid | magenta (255, 0, 255), `#FF00FF` | The **upper** eyelid: from the lid crease (the fold above the eye) down to the lid margin, with the upper lashes where they form a solid band. If no crease is visible, label the band of lid and lashes just above the eye opening. |
| 4 | caruncle | yellow (255, 255, 0), `#FFFF00` | The small pink mound in the inner corner of the eye (the nose side), with the pink fold next to it. |
| 5 | brow | red (255, 0, 0), `#FF0000` | Eyebrow hairs. Most VISOB crops show little or no brow. |
| 255 | ignore | grey (128, 128, 128), `#808080` | Only where you truly cannot decide, for example glare or blur hiding a boundary. Keep it under 5 % of the image. |

The lid definition follows the dataset the model was trained on (Nahass et al. 2025),
so that the model and the labels mean the same thing. The lower lid is skin in both.

Rules of thumb:

- Label what you see, not what you know is there. A covered part of the iris is not iris.
- Put boundaries on the visible edge. Aim for 1-2 pixels in the 240 x 160 image.
- Do not trace single lashes that cross the iris or sclera: those pixels stay iris or sclera.
- Reflections belong to whatever they sit on (iris, pupil or sclera).
- Eyeliner on the upper lid margin is lid; on the lower lid it is skin.
- A closed or half-closed eye is fine: label what is visible, even if there is no iris.
- Paint over the model's mistakes. Common ones: red "brow" blobs in the image corners,
  lid painted over the top of the iris, a missing caruncle, sclera leaking onto the lower lid.

## Option A (recommended): an image editor with layers

GIMP and Krita are free, run offline and need no server. For 60 images this is the
simplest route.

GIMP, for a prefilled image:

1. Open `images/<name>` (File > Open).
2. Add the prefill as a layer: File > Open as Layers > `prefill/<name>`.
3. In the Layers panel, set the prefill layer's opacity to about 50 % so you can see the eye.
4. Paint with the **Pencil** tool, not the Paintbrush (the pencil has hard edges). Set the
   colour by typing the hex code from the table into the colour dialog.
5. For large areas, use Free Select (lasso) with "Antialiasing" and "Feather edges"
   turned off, then Edit > Fill with FG Color.
6. When finished, set the prefill layer back to 100 % opacity and hide the photo layer
   (eye icon). Then File > Export As > `labels/<name>`, format PNG.

For a blind image, add a new layer filled with black instead of step 2 (Layer > New
Layer, fill type foreground colour, with black as the foreground), set it to 50 %
opacity and paint all classes on it.

Krita works the same way: use a pixel-art brush (no anti-aliasing), fill tools with
anti-aliasing off, and export only the mask layer as PNG.

## Option B: CVAT, installed on the laptop

CVAT has polygon and brush tools and can import the prefill. It runs in Docker on your
own machine.

1. Install Docker Desktop. Clone github.com/cvat-ai/cvat at a release tag, run
   `docker compose up -d`, create a user with
   `docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'`,
   and open http://localhost:8080.
2. Create a project with the labels `iris`, `sclera`, `lid`, `caruncle`, `brow` and
   `ignore`, with the exact colours from the table. Skin needs no label: unlabelled
   pixels are exported as black, which is skin.
3. Create two tasks: one with the prefilled images and one with the blind images, so
   nobody sees model masks for the blind ones.
4. In the prefilled task: Actions > Upload annotations > "Segmentation mask 1.1" >
   `cvat_prefill.zip`. This import has not been tested with a real CVAT install yet; if
   it fails, label with the overlays open next to CVAT instead.
5. When finished: Actions > Export task dataset > "Segmentation mask 1.1". Unzip it and
   copy the files in `SegmentationClass/` into `labels/` (they already have the right names).

## Option C: Label Studio, installed on the laptop

Label Studio installs with `pip install label-studio` (use a separate environment) and
runs at http://localhost:8080 after `label-studio start`. Drawing works, but it cannot
read our prefill directly, and its PNG export writes one file per class. We have no
converter for that, so prefer option A or B.

## Time and workload

These are estimates, not measurements. Time your first five images and tell the team.

- Correcting a prefilled image: about 5-10 minutes.
- Labelling a blind image from scratch: about 10-20 minutes.
- A pack of 60 (48 prefilled, 12 blind): about 6-10 person-hours. Split it three ways:
  20 images each, 4 of them blind.
- Agreement check: each person also labels 4 images that someone else labelled, into a
  separate folder `labels_b/`. The agreement between two people is the best score any
  model can be expected to reach.

## Quality rules

- Same file name as the image, PNG, 240 x 160 pixels, only the colours in the table
  (or the ids, as a grey-level PNG). At most 5 % ignore.
- After every 10 images, check your files (from the repository root, with the pack in `label_pack/`):
  `python scripts/18_validate_masks.py --labels-dir label_pack/labels --images-dir label_pack/images --check-only`.
  It lists files with the wrong size, colours outside the palette (usually anti-aliasing)
  or too much ignore. Off-palette pixels are counted as ignore.
- A second person looks over each finished label (about 2 minutes per image) for gross
  mistakes: lid painted as iris, a missed caruncle, a stray brow blob.
- Keep a small log, `labelling_log.csv`: name, annotator, minutes, notes.
- Never change labels after seeing validation scores.

## Run the validation

On the laptop, from the repository root, with the pack in `label_pack/`:

```
python scripts/18_validate_masks.py --labels-dir label_pack/labels \
    --labels-dir-b label_pack/labels_b --images-dir label_pack/images \
    --masks label_pack/model_masks_sample.npz --sample-csv label_pack/sample.csv \
    --out-dir validation --tag v1
```

No model and no VISOB copy are needed: everything is in the pack. The script writes:

- `label_qc_v1.csv`: one row per label file, with problems flagged.
- `validation_v1.csv`: IoU and Dice per image, method and class.
- `validation_summary_v1.csv`: mean IoU and Dice with 95 % subject-cluster bootstrap
  confidence intervals, and the pooled IoU, for all images, prefilled images and blind images.
- `validation_v1.md`: the tables in readable form.

How to read the results:

- Scores are computed at 256 x 256, the size the recognition model sees, because that
  is where region mass is measured. Hand labels are resized there the same way the images are.
- The learned masks are scored on all six classes. Learned and heuristic masks are both
  scored on three regions: iris (pupil included), sclera and periocular (everything else),
  because the heuristic masks have no lid, caruncle or brow.
- If the blind images score clearly lower than the prefilled ones, the prefill biased the
  labels: quote the blind numbers.
- Compare with the two-annotator agreement: a model cannot be expected to beat it.
- Agree on a pass mark **before** looking at the results. A reasonable one is what the
  model reaches on its own test data (Nahass et al. 2025: Dice 0.93 for the iris and 0.81
  for the sclera, which is IoU 0.87 and 0.68): for example, iris IoU at least 0.80 and
  sclera IoU at least 0.65 on the blind images. If the masks fail it, they need
  fine-tuning on our labels before region results can be reported.
