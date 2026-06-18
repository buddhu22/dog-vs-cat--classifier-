import os, sys
sys.path.insert(0, '.')
import tensorflow as tf
import cv2
import numpy as np

# 1. Dataset structure check
train_dir = './catsvsdogs/train'
test_dir = './catsvsdogs/test'

print('=== DATASET STRUCTURE ===')
if os.path.exists(train_dir):
    classes = sorted(os.listdir(train_dir))
    print(f'Train subdirectories: {classes}')
    for c in classes:
        d = os.path.join(train_dir, c)
        if os.path.isdir(d):
            count = len(os.listdir(d))
            print(f'  {c}: {count} files')

if os.path.exists(test_dir):
    classes = sorted(os.listdir(test_dir))
    print(f'Test subdirectories: {classes}')
    for c in classes:
        d = os.path.join(test_dir, c)
        if os.path.isdir(d):
            count = len(os.listdir(d))
            print(f'  {c}: {count} files')

# 2. Check class_names from image_dataset_from_directory
print()
print('=== CLASS NAMES FROM image_dataset_from_directory ===')
ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, labels='inferred', label_mode='int',
    batch_size=32, image_size=(256,256), shuffle=False
)
print(f'class_names: {ds.class_names}')
print(f'  label 0 = {ds.class_names[0]}')
print(f'  label 1 = {ds.class_names[1]}')

# 3. Check model
print()
print('=== MODEL CHECK ===')
model_path = './models/best_model.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f'Model input shape: {model.input_shape}')
    print(f'Model output shape: {model.output_shape}')
    print(f'Model built: {model.built}')
    
    for layer in model.layers:
        print(f'  {layer.name}: {layer.__class__.__name__}')
    
    # 4. Live inference test
    print()
    print('=== LIVE INFERENCE TEST ===')
    cat_dir = os.path.join(test_dir, 'cats')
    dog_dir = os.path.join(test_dir, 'dogs')
    
    for label_name, dirpath in [('cats', cat_dir), ('dogs', dog_dir)]:
        if os.path.isdir(dirpath):
            files = os.listdir(dirpath)[:3]
            for f in files:
                fpath = os.path.join(dirpath, f)
                img = cv2.imread(fpath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                img = img / 255.0
                img = np.expand_dims(img, 0)
                pred = model.predict(img, verbose=0)[0][0]
                cls = 'Dog' if pred >= 0.5 else 'Cat'
                print(f'  {label_name}/{f}: raw_prob={pred:.6f}, predicted={cls}')
    
    # 5. Test Grad-CAM model creation
    print()
    print('=== GRAD-CAM MODEL TEST ===')
    try:
        # Find last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        print(f'Last Conv2D layer: {last_conv}')
        
        # Try to build grad model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv).output, model.output]
        )
        print('Grad-CAM sub-model created successfully')
        
        # Try a forward pass
        dummy = np.random.rand(1, 256, 256, 3).astype(np.float32)
        conv_out, pred_out = grad_model(dummy)
        print(f'Conv output shape: {conv_out.shape}')
        print(f'Pred output shape: {pred_out.shape}')
    except Exception as e:
        print(f'Grad-CAM ERROR: {e}')
        import traceback
        traceback.print_exc()
else:
    print('Model not found!')
