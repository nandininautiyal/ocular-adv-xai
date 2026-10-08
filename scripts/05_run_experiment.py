"""
scripts/05_run_experiment.py

End-to-end orchestration for the Ocular Adversarial XAI project.
Requires the trained model checkpoint (.keras) to run.

Workflow for each test image:
1. Load image
2. Clean baseline: predict & generate XAI heatmap
3. Attack: generate adversarial image & adversarial heatmap
4. Defend: apply inference-time defence & defended heatmap
5. Compare: calculate divergence (SSIM, L2, etc.) between heatmaps
"""

import os
import argparse
import json

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.xai.attribution import layercam
from src.attacks.whitebox import ATTACK_REGISTRY
from src.defences.inference import DEFENCE_REGISTRY
from src.compare.divergence import compare_heatmaps

# Handle ArcFace custom object for loading
from third_party.ABRobOcular.train import ArcFace


def run_experiment(model_path, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'ArcFace': ArcFace}
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Load & Preprocess Image
    # Assuming standard image loading for now, (256, 256, 3) float32 [0,1]
    img_raw = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img = tf.keras.preprocessing.image.img_to_array(img_raw) / 255.0
    img = img.astype(np.float32)

    # 3. Clean Baseline
    print("Computing clean baseline...")
    dummy_label = np.zeros((1, 300), dtype=np.float32)
    clean_pred = model.predict([np.expand_dims(img, 0), dummy_label], verbose=0)
    clean_class = int(np.argmax(clean_pred[0]))
    
    clean_heatmap = layercam(model, img, class_idx=clean_class)
    
    results = {
        "image_path": image_path,
        "clean_class": clean_class,
        "experiments": []
    }

    # 4. Attacks
    for attack_name, attack_fn in ATTACK_REGISTRY.items():
        print(f"Running attack: {attack_name}...")
        adv_img = attack_fn(img, model)
        
        adv_pred = model.predict([np.expand_dims(adv_img, 0), dummy_label], verbose=0)
        adv_class = int(np.argmax(adv_pred[0]))
        adv_heatmap = layercam(model, adv_img, class_idx=adv_class)
        
        # 5. Defences
        for def_name, def_fn in DEFENCE_REGISTRY.items():
            if def_name == "strip":
                continue # Skip strip for now as it requires overlay pool
                
            print(f"  Running defence: {def_name}...")
            def_img = def_fn(adv_img)
            
            def_pred = model.predict([np.expand_dims(def_img, 0), dummy_label], verbose=0)
            def_class = int(np.argmax(def_pred[0]))
            def_heatmap = layercam(model, def_img, class_idx=def_class)
            
            # 6. Compare
            divergence = compare_heatmaps(clean_heatmap, adv_heatmap, def_heatmap)
            
            results["experiments"].append({
                "attack": attack_name,
                "defence": def_name,
                "adv_class": adv_class,
                "def_class": def_class,
                "divergence": divergence
            })

    # 7. Save results
    out_file = os.path.join(output_dir, "results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Experiment complete. Results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    run_experiment(args.model, args.image, args.out)
