# 🎬 ALNS Optimization Video Creation Guide

## Overview
This guide shows how to create animated videos of your ALNS optimization process, allowing you to visualize how routes evolve and improve over iterations.

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install matplotlib numpy imageio Pillow
# Optional for MP4 creation:
pip install ffmpeg-python moviepy
```

### 2. Basic Video Creation
```python
from optimization_video_creator import OptimizationVideoCreator

# Initialize video creator
creator = OptimizationVideoCreator()

# Your optimization history (list of states per iteration)
optimization_history = [
    {
        'iteration': 1,
        'cost': 100.0,
        'best_cost': 100.0,
        'routes': [route1_coords, route2_coords]
    },
    # ... more iterations
]

# Create video
video_path = creator.create_optimization_animation(
    optimization_history=optimization_history,
    customer_data=customer_locations,
    depot_location=(0, 0),
    intermediate_facilities=[(10, 10)],
    output_filename="my_optimization.mp4"
)
```

## 📊 Video Types Available

### 1. **Route Animation** (`create_optimization_animation`)
- Shows routes being built/optimized step by step
- Displays current cost, iteration number, route details
- Color-coded routes with direction arrows

### 2. **Cost Convergence** (`create_cost_convergence_video`)
- Shows how total cost improves over iterations
- Animated cost curve with current point highlighted
- Great for showing algorithm convergence

### 3. **Side-by-Side Comparison** (`create_side_by_side_video`)
- Combines route visualization and cost graph
- Shows both spatial and temporal optimization progress
- Perfect for presentations and analysis

## 🔧 Integration with Your Existing ALNS

### Option 1: Modify Your ALNS Class
```python
class ALNSOptimizer:
    def __init__(self, problem_data):
        self.problem_data = problem_data
        self.optimization_history = []  # Add this
        
    def optimize(self, max_iterations):
        for iteration in range(max_iterations):
            # Your existing optimization logic
            
            # Add tracking every few iterations
            if iteration % 10 == 0:
                state = {
                    'iteration': iteration,
                    'cost': current_cost,
                    'best_cost': best_cost_sofar,
                    'routes': self._extract_route_coordinates(current_solution)
                }
                self.optimization_history.append(state)
        
        return self.optimization_history
```

### Option 2: Use the ALNSWithVideoTracking Class
```python
from alns_video_integration import ALNSWithVideoTracking

# Replace your existing ALNS initialization
alns = ALNSWithVideoTracking(problem_data)

# Run optimization with tracking
final_solution = alns.run_alns(max_iterations=100)

# Create video automatically
video_path = alns.create_video_from_history(customer_data)
```

## 📋 Data Format Requirements

### Optimization History Structure
```python
optimization_history = [
    {
        'iteration': 1,
        'cost': 45.67,
        'best_cost': 45.67,
        'routes': [
            [(0, 0), (3, 15), (7, 12), (0, 0)],  # Route 1 coordinates
            [(0, 0), (15, 5), (18, 15), (0, 0)]   # Route 2 coordinates
        ]
    }
    # ... more iterations
]
```

### Customer Data Format
```python
customer_data = {
    (3, 15): 4,   # (x, y): demand
    (7, 12): 4,
    (5, 8): 5,
    # ... more customers
}
```

## 🎯 Advanced Usage

### Custom Video Settings
```python
creator = OptimizationVideoCreator()

# Adjust frame rate (slower = more detailed)
creator.fps = 1  # 1 frame per second (slow, detailed)

# Custom video quality
creator.dpi = 200  # Higher DPI for better quality

# Create multiple videos with different settings
route_video = creator.create_optimization_animation(...)
cost_video = creator.create_cost_convergence_video(...)
comparison_video = creator.create_side_by_side_video(...)
```

### Adding Custom Metrics to Video
```python
def create_custom_video(optimization_history, customer_data):
    creator = OptimizationVideoCreator()
    
    # Extend the _plot_metrics method to show custom metrics
    def custom_metrics_plot(ax, state, frame, total_frames):
        # Your custom metrics
        custom_text = f"""
        Custom Metric 1: {state.get('custom_metric', 'N/A')}
        Custom Metric 2: {state.get('another_metric', 'N/A')}
        """
        # ... implement custom plotting
        
    creator._plot_metrics = custom_metrics_plot
    return creator.create_optimization_animation(...)
```

## 🎨 Visualization Customization

### Colors and Styles
```python
# Modify _plot_routes method for custom colors
def custom_route_colors():
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    # Your custom color scheme
```

### Adding Animation Effects
```python
# The video creator automatically handles:
# - Route path animations
# - Direction arrows
# - Color transitions
# - Text updates

# For advanced effects, modify the animate_frame function
```

## 📁 Output Files

### Video Formats
- **MP4**: Best quality, requires ffmpeg
- **GIF**: Universal compatibility, larger file size
- **AVI**: Alternative format option

### File Organization
```
optimization_videos/
├── alns_optimization_process.mp4
├── cost_convergence.mp4
└── optimization_comparison.mp4
```

## ⚡ Performance Tips

### Efficient History Tracking
```python
# Don't track every iteration (creates too many frames)
if iteration % 5 == 0:  # Track every 5th iteration
    self.optimization_history.append(state)

# Or track based on improvement
if abs(last_tracked_cost - current_cost) > threshold:
    self.optimization_history.append(state)
```

### Memory Management
```python
# Limit history size for long runs
if len(self.optimization_history) > 200:
    self.optimization_history = self.optimization_history[-100:]  # Keep last 100
```

## 🎓 Academic/Presentation Use

### Creating Publication-Ready Videos
1. Use high DPI (150-300)
2. Consistent color scheme
3. Clear legends and labels
4. Appropriate frame rate (1-2 fps)
5. Include iteration numbers and cost metrics

### Presentation Tips
- Side-by-side videos work great for talks
- Cost convergence videos show algorithm effectiveness
- Route animations demonstrate problem complexity
- Consider creating 30-60 second videos for presentations

## 🔧 Troubleshooting

### Common Issues

**FFmpeg not found:**
```python
# System will automatically fallback to GIF
print("Note: Install ffmpeg for MP4 creation")
```

**Large file sizes:**
```python
# Reduce resolution
creator.dpi = 100

# Increase frame interval
creator.fps = 0.5
```

**Memory issues:**
```python
# Reduce history length
optimization_history = optimization_history[::2]  # Every 2nd frame
```

## 📈 Integration Examples

### With Your Current Project
1. Import `optimization_video_creator.py` into your project
2. Add history tracking to your ALNS loop
3. Create videos at the end of optimization
4. Perfect for showing algorithm progress in reports!

### Real-World Usage
- **Research Papers**: Show algorithm evolution
- **Presentations**: Demonstrate optimization progress
- **Debugging**: Visualize why algorithm gets stuck
- **Reports**: Professional animated explanations

## 🎉 Success!

With these tools, you can create professional-quality videos of your ALNS optimization process, perfect for academic papers, presentations, and understanding how your algorithm works!