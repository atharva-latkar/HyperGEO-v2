"""
Export utilities — save results to GeoTIFF, PNG, CSV report.
"""

import numpy as np
import os
from pathlib import Path


def export_color_map_png(color_map: np.ndarray, filepath: str):
    """Save the RGB mineral map as PNG."""
    try:
        from PIL import Image
        img = Image.fromarray(color_map, 'RGB')
        img.save(filepath)
        return True
    except ImportError:
        # Fallback: matplotlib
        import matplotlib.pyplot as plt
        plt.imsave(filepath, color_map)
        return True


def export_label_map_tiff(label_map: np.ndarray, filepath: str,
                           transform=None, crs=None):
    """Save integer label map as GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        rows, cols = label_map.shape
        t = transform or from_bounds(0, 0, cols, rows, cols, rows)
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=rows, width=cols,
            count=1,
            dtype=label_map.dtype,
            crs=crs,
            transform=t
        ) as dst:
            dst.write(label_map, 1)
        return True
    except ImportError:
        return False


def export_report_csv(result, output_path: str):
    """Save per-class pixel count and percentage to CSV."""
    import csv
    rows, cols = result.label_map.shape
    total = rows * cols
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mineral', 'Category', 'Pixels', 'Percentage'])
        for idx, name in enumerate(result.label_names):
            count = int(np.sum(result.label_map == idx))
            pct = count / total * 100
            writer.writerow([name, '', count, f'{pct:.2f}%'])
    return True


def export_full_report(result, output_dir: str, base_name: str = 'result'):
    """Export color map PNG + CSV report to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'{base_name}_mineral_map.png')
    csv_path = os.path.join(output_dir, f'{base_name}_report.csv')

    color_map = result.color_map
    if color_map is not None:
        export_color_map_png(color_map, png_path)

    export_report_csv(result, csv_path)
    return png_path, csv_path
