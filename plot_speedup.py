import matplotlib.pyplot as plt

n = [256, 512, 1024, 1536, 2048]
speedup = [1.64, 1.53, 1.23, 1.10, 1.28]

plt.figure(figsize=(8,5))
plt.plot(n, speedup, marker='o', linestyle='-', linewidth=2, markersize=8, color='b')
plt.xlabel('Matrix size n', fontsize=12)
plt.ylabel('Speedup (Serial / SIMD)', fontsize=12)
plt.title('SIMD Speedup vs Matrix Size', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(n)
plt.ylim(0.8, 1.8)
for i, v in enumerate(speedup):
    plt.text(n[i], v+0.03, f'{v:.2f}', ha='center', fontsize=10)
plt.savefig('speedup.pdf', format='pdf')
plt.savefig('speedup.png', dpi=150)
print("Generated speedup.pdf and speedup.png")
