import matplotlib.pyplot as plt

classes = ['Person', 'Car', 'Bus', 'Bicycle']
ap_scores = [0.4160, 0.4352, 0.3778, 0.3465]
colours = ['red', 'blue', 'green', 'orange']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(classes, ap_scores, color=colours, width=0.6)

for bar, score in zip(bars, ap_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

ax.set_ylabel('AP @ IoU=0.50:0.95', fontsize=12)
ax.set_title('Per-Class AP — Best Model (VOC2012 Gentle Aug + Soft-NMS)', fontsize=13)
ax.set_ylim(0, 0.55)

plt.tight_layout()
plt.savefig('perclass_ap.png', dpi=200)
print('Saved to perclass_ap.png')