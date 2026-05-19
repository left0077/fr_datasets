"""Show sample face pairs from CFP-FP dataset."""
import numpy as np
import pickle
import io
from PIL import Image
import matplotlib.pyplot as plt

# Load
with open('cfp_fp.bin', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

images = data[0]  # 14000 JPEG byte arrays
labels = data[1]  # 7000 bool labels (True=same person, False=different)

def decode_img(raw):
    return Image.open(io.BytesIO(raw.tobytes()))

# Show 3 "same person" pairs and 3 "different person" pairs
fig, axes = plt.subplots(4, 3, figsize=(8, 10))

same_idxs = [i for i, l in enumerate(labels) if l][:3]
diff_idxs = [i for i, l in enumerate(labels) if not l][:3]

for col, pair_idx in enumerate(same_idxs):
    img0 = decode_img(images[pair_idx * 2])
    img1 = decode_img(images[pair_idx * 2 + 1])
    axes[0, col].imshow(img0)
    axes[0, col].set_title(f'Frontal (#{pair_idx})', fontsize=9)
    axes[0, col].axis('off')
    axes[1, col].imshow(img1)
    axes[1, col].set_title(f'Profile (#{pair_idx})', fontsize=9)
    axes[1, col].axis('off')

for col, pair_idx in enumerate(diff_idxs):
    img0 = decode_img(images[pair_idx * 2])
    img1 = decode_img(images[pair_idx * 2 + 1])
    axes[2, col].imshow(img0)
    axes[2, col].set_title(f'Frontal (#{pair_idx})', fontsize=9)
    axes[2, col].axis('off')
    axes[3, col].imshow(img1)
    axes[3, col].set_title(f'Profile (#{pair_idx})', fontsize=9)
    axes[3, col].axis('off')

# Add row labels
for ax, label in zip(axes[:, 0], ['Same Person\n(Frontal)', 'Same Person\n(Profile)',
                                    'Different Person\n(Frontal)', 'Different Person\n(Profile)']):
    ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=50, va='center')

plt.suptitle('CFP-FP Dataset Samples', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('cfp_samples.png', dpi=120, bbox_inches='tight')
print('Saved to cfp_samples.png')
