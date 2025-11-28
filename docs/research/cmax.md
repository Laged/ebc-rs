# Secrets of Edge-Informed Contrast Maximization for Event-Based Vision (WACV 2025)

## Executive Summary

This paper introduces a novel "bi-modal" framework for Contrast Maximization (CM) in event-based vision. Traditional CM optimizes motion parameters by maximizing the sharpness (contrast) of the Image of Warped Events (IWE) alone. This paper proposes adding a second objective: **maximizing the correlation between the IWE and a sharp Edge Map derived from a standard frame**.

**Key Insight for ebc-rs**: By guiding the CM optimization with a known sharp edge map (even a proxy one), we can improve convergence and accuracy, especially in textureless regions where pure event contrast might be ambiguous.

## Core Concepts

### 1. The Bi-Modal Objective Function

The standard CM objective maximizes the contrast of the IWE:
$$J_{contrast}(\theta) = \text{Contrast}(IWE(\theta))$$

The proposed **Edge-Informed** objective adds a correlation term:
$$J_{total}(\theta) = J_{contrast}(\theta) + \lambda \cdot J_{correlation}(IWE(\theta), E_{ref})$$

Where:
- $\theta$: Motion parameters (e.g., angular velocity $\omega$).
- $IWE(\theta)$: Image of Warped Events.
- $E_{ref}$: Reference Edge Map (e.g., Canny edges from a frame).
- $\lambda$: Weighting factor.

### 2. Reference Time Alignment

A critical detail is that the IWE and the Edge Map must be aligned in time.
- **Edge Map ($E_{ref}$)**: Extracted from a frame at time $t_{frame}$.
- **Event Warping**: Events must be warped to $t_{ref} = t_{frame}$.

If $t_{ref}$ is mismatched (e.g., set to the middle of the window), the IWE will be spatially offset from the Edge Map, breaking the correlation.

### 3. Edge Extraction Pipeline (Paper)

The paper uses a sophisticated pipeline to generate $E_{ref}$ from grayscale frames:
1.  **Denoising**: Non-local means.
2.  **Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization).
3.  **Sharpening**: Gaussian sharpening.
4.  **Edge Detection**: Canny edge detector.
5.  **Smoothing**: Gaussian blur (to widen the basin of attraction for correlation).

## Algorithm Steps

1.  **Input**: Event stream $\mathcal{E}$ and Reference Frame $\mathcal{I}$.
2.  **Edge Map Generation**: Compute $E_{ref}$ from $\mathcal{I}$ (Canny + Blur).
3.  **Optimization Loop**:
    a.  **Warp**: Warp events to $t_{ref}$ using current $\theta$.
    b.  **IWE**: Accumulate warped events into $IWE(\theta)$.
    c.  **Contrast**: Compute variance/gradient magnitude of $IWE(\theta)$.
    d.  **Correlation**: Compute pixel-wise product $IWE(\theta) \cdot E_{ref}$.
    e.  **Update**: Adjust $\theta$ to maximize the combined score.

## Relevance to ebc-rs (Fan RPM Estimation)

### Adaptation for Event-Only Data
Since we do not have frames, we must adapt the "Edge-Informed" concept:
- **Proxy Reference**: Instead of a frame, we use a "Short-Window Accumulation" of events at the end of the window.
- **Assumption**: Over a very short window (e.g., 1ms), motion blur is negligible, so the accumulation approximates a sharp frame.
- **Pipeline**:
    1.  Accumulate events in $[t_{end} - \delta, t_{end}]$ -> `ShortWindowSurfaceImage`.
    2.  Compute Sobel edges on this image -> `ReferenceEdgeMap`.
    3.  Warp full window events $[t_{start}, t_{end}]$ to $t_{ref} = t_{end}$.
    4.  Optimize $\omega$ to maximize Contrast + Correlation with `ReferenceEdgeMap`.

### Implementation Details
- **GPU**: Both Contrast and Correlation can be computed efficiently in a compute shader.
- **Correlation Kernel**: Simple dot product (pixel-wise multiplication) summed over the image.
- **Challenges**: The "Short-Window" reference might be noisy. Denoising (as done in the paper) is important.

## References
- [Paper (CVF Open Access)](https://openaccess.thecvf.com/content/WACV2025/papers/Karmokar_Secrets_of_Edge-Informed_Contrast_Maximization_for_Event-Based_Vision_WACV_2025_paper.pdf)
