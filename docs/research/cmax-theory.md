
Event-Based Vision Architectures for High-Speed Rotational Metrology: A Monograph on Contrast Maximization, Edge Detection, and Systems Implementation


1. Executive Summary

The transition from frame-based acquisition to event-based sensing represents one of the most significant architectural shifts in the history of computer vision, particularly within the domain of high-speed industrial metrology. This research report provides an exhaustive analysis of utilizing neuromorphic vision sensors—commonly known as event cameras—for the precise estimation of rotational dynamics, specifically the Revolutions Per Minute (RPM) of fast-spinning fans and propellers. Unlike traditional sensors that integrate photon flux over a fixed exposure time, thereby succumbing to motion blur and temporal aliasing, event cameras operate asynchronously, generating microsecond-resolution data packets in response to logarithmic intensity changes. This fundamental operational difference necessitates a complete reimagining of image processing pipelines, moving away from static frame analysis toward continuous-time optimization frameworks.
The core of this analysis is the Contrast Maximization (CMAX) framework, a rigorously derived optimization method that recovers motion parameters by maximizing the sharpness of a motion-compensated "Image of Warped Events" (IWE). We posit that in the context of high-speed rotation, CMAX is not merely a motion estimation technique but a robust, spatiotemporal form of edge detection that converts temporal noise into geometric signal. By "focusing" the asynchronous event stream onto a reference time plane, CMAX reveals the latent edge structure of the rotating object, allowing for precise RPM calculation even in challenging illumination conditions.
This document dissects the theoretical underpinnings of CMAX, establishing the mathematical relationships between rotational warping functions, objective functions (such as Variance, Mean Squared Magnitude, and Entropy), and the phenomenon of "Event Collapse." We critically evaluate recent algorithmic advancements, including the "EventPro" and "EV-Tach" systems, which introduce geometry-instructed constraints to regularize the optimization landscape. Furthermore, we conduct a rigorous comparative analysis of implementation architectures, contrasting the emerging Rust ecosystem (leveraging crates such as nalgebra, davis-EDI-rs, and wgpu) against established GPU (CUDA) pipelines. Our findings suggest that while GPU architectures offer theoretical throughput advantages for massive batch processing, the Rust ecosystem provides superior safety, determinism, and lower latency for the specific scalar optimization problems found in rotational tachometry. This report serves as a definitive technical guide for architects and researchers designing next-generation, event-based perception systems.

2. The Physics and Mathematics of Event Generation in Rotational Scenarios

To understand the efficacy of Contrast Maximization for RPM estimation, one must first deeply understand the signal generation process of the event sensor itself, particularly how it couples with the physics of rotation.

2.1 The Asynchronous Pixel Model

Standard cameras operate on a synchronous, global shutter paradigm. They integrate light intensity $I(u, v, t)$ over an exposure time $\Delta t_{exp}$ and output a matrix of values at fixed intervals $t_k$. This introduces two fatal flaws for high-speed metrology:
Motion Blur: If the angular velocity $\omega$ of a fan blade is high, the blade traverses multiple pixels during $\Delta t_{exp}$. The resulting image $I_{frame}$ is a temporal average, smearing high-frequency spatial details (edges) and making precise localization impossible.1
Temporal Aliasing: The Nyquist-Shannon sampling theorem dictates that to resolve a signal of frequency $f_{max}$, the sampling rate $f_s$ must be $> 2f_{max}$. For a drone propeller spinning at 10,000 RPM ($\approx 166$ Hz) with multiple blades, the visual frequency can exceed 1 kHz. A standard 60 fps camera undersamples this signal by orders of magnitude, leading to the "wagon-wheel effect" where the fan appears to spin backward or stand still.3
Event cameras (e.g., Dynamic Vision Sensors, DVS) fundamentally reject this integration model. Each pixel contains an analog photoreceptor and a differential circuit. It monitors the log-intensity $L(u, v, t) = \ln(I(u, v, t))$. An event $e_k$ is emitted if and only if:


$$|L(u, v, t) - L(u, v, t_{last})| \geq C$$

where $C$ is the contrast threshold (typically 10-15%) and $t_{last}$ is the timestamp of the previous event at that pixel.
The output is a continuous, asynchronous stream of events:


$$\mathcal{E} = \{e_k\}_{k=1}^N = \{(x_k, y_k, t_k, p_k)\}$$

where $p_k \in \{+1, -1\}$ indicates the polarity of the change (brightness increase or decrease).1

2.2 The Geometry of Rotation in Space-Time

Consider a fan blade rotating around a center $\mathbf{c} = (c_x, c_y)$ with angular velocity $\omega$. In the 3D space-time volume spanned by the sensor $(x, y, t)$, the blade does not trace a smeared path. Instead, the edges of the blade trace a coherent, continuous manifold—a helix.5
For a point $\mathbf{p}$ on the blade at radius $r$, its position at time $t$ is:
$$ \mathbf{p}(t) = \mathbf{c} + r \begin{bmatrix} \cos(\omega t + \phi_0) \ \sin(\omega t + \phi_0) \end{bmatrix} $$
Events are generated strictly at the boundaries of the blade where the spatial gradient $\nabla I$ is non-zero and moving. The event generation rate at a pixel is given by the optical flow equation:


$$\frac{\partial L}{\partial t} = -(\nabla L \cdot \mathbf{v})$$

where $\mathbf{v}$ is the velocity field. Since $\mathbf{v} = \boldsymbol{\omega} \times \mathbf{r}$, the event rate is maximal at the edges of the blade moving perpendicular to the gradient. This means the event stream is inherently a sparse, edge-encoded representation of the rotating object.6
The task of RPM calculation, therefore, is not "tracking" in the conventional sense. It is a problem of geometric rectification. We seek the parameter $\omega$ that, when applied to "unwind" the space-time helix, flattens all events onto a single 2D plane (the reference time $t_{ref}$), reconstructing the sharp edge map of the static fan.

3. The Contrast Maximization (CMAX) Framework

Contrast Maximization is the state-of-the-art framework for event-based motion estimation. It unifies tracking, optical flow, and rotational estimation under a single energy-maximization principle.

3.1 The Warping Function $\mathcal{W}$

The core mechanism of CMAX is the warping function $\mathcal{W}(\mathbf{x}, t; \theta)$, which maps an event's location $\mathbf{x} = (x, y)$ at time $t$ to a reference location $\mathbf{x}'$ at time $t_{ref}$, based on a candidate motion parameter vector $\theta$.
For the specific case of calculating RPM, the motion model is a planar Euclidean rotation around a center $\mathbf{c}$. The parameter vector is $\theta = \{\omega, \mathbf{c}\}$. If we assume the center $\mathbf{c}$ is known (or estimated separately), the problem becomes a 1-DoF search for $\omega$.
The warp equation for the $k$-th event is:
$$ \mathbf{x}'_k = \mathcal{W}(\mathbf{x}k, t_k; \omega, \mathbf{c}) = R(\omega(t{ref} - t_k)) (\mathbf{x}_k - \mathbf{c}) + \mathbf{c}
$$where $R(\phi)$ is the 2D rotation matrix:$$
R(\phi) = \begin{bmatrix} \cos\phi & -\sin\phi \ \sin\phi & \cos\phi \end{bmatrix} $$
Implementation Note: This seemingly simple equation involves a heavy computational load. For a stream of $10^6$ events/second, evaluating this warp requires millions of trigonometric operations. Optimization strategies often involve:
Linearization: Assuming small angles for short time windows, $\cos \phi \approx 1$ and $\sin \phi \approx \phi$. However, for high-speed fans (e.g., 10,000 RPM), $\Delta \theta$ can be large even in milliseconds, making linearization risky.6
Look-up Tables: Pre-computing sine/cosine values.
Lie Algebra: Operating in the Lie algebra $\mathfrak{so}(2)$ (skew-symmetric matrices) and using the exponential map, although for 2D rotation this simplifies to complex number multiplication.8

3.2 The Image of Warped Events (IWE)

Once the events are warped to $\mathbf{x}'_k$, they are accumulated into a discretized image grid called the Image of Warped Events (IWE) or $H$.


$$H(\mathbf{u}; \theta) = \sum_{k=1}^N \delta(\mathbf{u} - \mathbf{x}'_k)$$

In practice, $\delta$ is replaced by a bilinear interpolation kernel or a Gaussian kernel to ensure the objective function is differentiable with respect to $\theta$.9

$$H(\mathbf{u}) = \sum_{k} b(\mathbf{u} - \mathbf{x}'_k)$$

where $b(\cdot)$ is the bilinear interpolation weight.
Interpretation:
Correct $\omega$: Events generated by the same physical edge of the fan blade at different times $t_k$ will be warped to the exact same position $\mathbf{x}'$. They will "stack up" in the IWE, creating pixels with very high values (peaks) against a background of near-zero values. The image will appear sharp and high-contrast.
Incorrect $\omega$: Events will map to different locations. The edge will be "smeared" across the IWE. The histogram of pixel values will be unimodal and flat (low contrast).5

3.3 Objective Functions: The Mathematical Measures of Focus

The "sharpness" of the IWE must be quantified by a scalar objective function $\mathcal{L}(\theta)$. Several candidates exist in the literature, each with trade-offs regarding convexity, convergence basin, and computational cost.

3.3.1 Variance ($\mathcal{L}_{Var}$)

The most widely adopted metric is the variance of the IWE pixel values:


$$\mathcal{L}_{Var}(\theta) = \frac{1}{N_p} \sum_{i,j} (H(i,j) - \mu_H)^2$$

where $\mu_H$ is the mean event count per pixel.
Mechanism: Variance rewards "peakiness." A distribution with a few very high values and many zeros has higher variance than a uniform distribution.
Relation to Mean Square: Since $\mu_H = N / N_p$ is constant for a fixed set of events regardless of the warp, maximizing variance is mathematically equivalent to maximizing the Mean Squared (MS) magnitude:

$$\mathcal{L}_{MS}(\theta) = \sum_{i,j} H(i,j)^2$$

This simplification (omitting the mean subtraction) is crucial for efficient FPGA and GPU implementations.12
Basin of Attraction: Variance typically exhibits a smooth, convex basin of attraction near the true parameter, facilitating gradient-based or iterative search methods.11

3.3.2 Mean Squared Magnitude of the Gradient (MSG)

$$ \mathcal{L}{MSG}(\theta) = \sum{i,j} |
| \nabla H(i,j) ||^2 $$
This metric explicitly measures the strength of spatial edges in the IWE.
Pros: It can be more sensitive to fine alignment than simple variance, as it rewards spatial structure (edges) rather than just pixel pile-up.
Cons: Calculating the gradient $\nabla H$ (e.g., via Sobel operator) is computationally expensive, adding an $O(N_p)$ pass after every warp iteration.14

3.3.3 Entropy


$$\mathcal{L}_{Ent}(\theta) = - \sum_{k} p_k \ln(p_k)$$

where $p_k$ is the normalized histogram of the IWE. Minimizing entropy encourages the intensity distribution to be "sparse" (mostly zeros, some highs). However, entropy landscapes can be more non-convex and sensitive to noise than variance.15

3.4 The Phenomenon of Event Collapse

A critical pathology in CMAX is Event Collapse. This occurs when the warping function compresses the spatial domain of the events.
Scenario: Imagine a warp that scales all event coordinates to a single point $\mathbf{x}' = \mathbf{c}$.
Result: All events pile up at $\mathbf{c}$. The pixel at $\mathbf{c}$ has value $N$, and all others are $0$.
False Maximum: This configuration yields a massive Variance/Mean Square score, often higher than the "correct" sharp image.
Relevance to RPM: In rotational CMAX, pure rotation preserves area (determinant of rotation matrix is 1), so geometric collapse is less inherent than in optical flow (where $z \to 0$ causes collapse). However, if the center of rotation $\mathbf{c}$ is also being optimized, the algorithm might drift $\mathbf{c}$ to infinity or shrink the radius to maximize density.11
Mitigation: This necessitates Geometry-Instructed constraints (discussed in Section 4) or regularization terms that penalize deviations from area-preserving transformations.13

4. Advanced Algorithmic Frameworks: EventPro and EV-Tach

Recent literature has moved beyond generic CMAX to specialized algorithms for rotational metrology. We examine two leading frameworks: EV-Tach and EventPro.

4.1 EV-Tach: The Handheld Tachometer

EV-Tach 17 addresses the challenge of measuring RPM with a handheld event camera, where the camera's own ego-motion (jitter) adds noise to the rotational signal.

4.1.1 Algorithm Breakdown

Clustering Initialization: The algorithm first segments the event stream into clusters to identify the rotating object. It uses a Cluster-Centroids Initialization module to robustly find the center of rotation $\mathbf{c}$. This avoids the local minima problem where CMAX might latch onto a background edge.
Outlier Removal: Hand movements generate "background events." EV-Tach employs a statistical filter. It assumes the fan blades generate a high density of events in a circular annulus. Events falling outside this spatial distribution (e.g., in the center hub or far corners) are discarded based on their distance from the centroid.20
Coarse-to-Fine Alignment: Instead of a single expensive search, it performs:
Coarse Search: Steps of 100 RPM to find a global maximum basin.
Fine Search: Gradient descent or smaller steps within the basin to lock onto the precise frequency (precision up to 0.03%).21

4.2 EventPro: Geometry-Instructed Compensation

EventPro 19 represents the state-of-the-art in drone propeller sensing. It introduces the concept of "Count Every Rotation" by explicitly modeling the geometry of the propeller.

4.2.1 Hierarchical Preprocessing

EventPro acknowledges that raw event streams are noisy. It uses Distribution-Informed Filtering:
Temporal Binning: Events are binned into histograms.
Spatial Filtering: Isolated events (shot noise) are removed. Only events that have spatiotemporal neighbors (supporting the hypothesis of a moving edge) are retained.

4.2.2 Geometry-Instructed Objective Function

Unlike generic CMAX which blindly maximizes variance, EventPro adds a Sparsity-Aware Reward.23

$$R_{total} = R_{accumulation} + \lambda R_{sparsity}$$
Accumulation Reward ($R_{acc}$): Similar to Variance, it rewards high pixel counts.

$$R_{acc} = \sum_{i,j} \exp(H(i,j))$$

The exponential amplifies peaks more aggressively than the square, favoring very sharp alignments.
Sparsity Reward ($R_{sparsity}$): It penalizes the number of active pixels.

$$R_{sparsity} = - \sum_{i,j} \mathbb{1}(H(i,j) > 0)$$

This term combats blur directly. A blurred edge activates many pixels; a sharp edge activates few. This acts as a regularizer against "thick" edges, forcing the solution toward the thinnest possible blade representation.

4.2.3 Periodicity Constraints

EventPro leverages the fact that a propeller with $B$ blades has rotational symmetry. Ideally, warping by $\Delta \theta = 2\pi / B$ should result in self-similarity. This allows the algorithm to not just estimate speed, but also verify the blade count and detect damaged blades (asymmetry).19

5. Comparative Analysis of Implementation Architectures

The implementation of these algorithms requires navigating a complex landscape of hardware and software choices. The high data rate of event cameras (1-100 Million Events Per Second, MEPS) demands high-performance computing. We contrast the Rust ecosystem (CPU/GPU) against the traditional C++/CUDA ecosystem.

5.1 The Rust Ecosystem: Safety and Performance

Rust has emerged as a formidable language for neuromorphic engineering due to its unique blend of high-level abstractions and low-level memory control without the safety risks of C++.

5.1.1 Core Crates for Event Vision

davis-EDI-rs 26:
Significance: This is a reference implementation for high-speed event processing.
Architecture: It utilizes a lock-free ring buffer pattern to decouple the USB polling thread (producing events) from the reconstruction thread (consuming events). This prevents the "producer" from blocking the "consumer," ensuring no events are dropped during intense computation.
Relevance to RPM: This architecture is ideal for a real-time tachometer. The optimization loop can run continuously on the latest snapshot of the ring buffer.
aedat-rs 28:
Function: A highly optimized parser for the AEDAT4 file format (used by iniVation/DAVIS cameras).
Performance: Written using nom or direct binary parsing, it outperforms Python-based decoders by orders of magnitude. Efficient I/O is critical; if the system spends 50% of its time just parsing timestamps, the real-time constraint is violated.
nalgebra 30:
Function: General-purpose linear algebra.
Usage in CMAX: It provides Vector2<f32>, Rotation2<f32>, and Complex<f32> types.
Optimization: nalgebra is no_std compatible (good for embedded) and supports SIMD optimizations. Rotations in 2D can be implemented via complex multiplication, which is often faster than matrix multiplication.
Example: let rot = Rotation2::new(omega * dt); let new_pos = rot * (pos - center) + center;

5.1.2 GPU Acceleration with wgpu

For scenarios where CPU throughput is insufficient, Rust offers wgpu 32, a safe, portable WebGPU implementation.
Compute Shaders: CMAX is essentially a "scatter" operation. Events are scattered onto the IWE grid.
The Atomic Bottleneck:
In wgpu WGSL shaders, accumulation is done via atomicAdd(&buffer[index], 1u).
The Hotspot Problem: For a spinning fan, the center of rotation and the blade edges are extremely dense with events. Thousands of GPU threads trying to atomicAdd to the same few pixels create massive memory contention.
Solution: Use Shared Memory (Workgroup Memory). Each workgroup (e.g., 256 threads) accumulates into a fast local buffer (L1 cache), and then flushes to global memory only once. This is a classic histogram optimization pattern.34
Rust Integration: wgpu allows the host (Rust) to manage the optimization loop (Brent's method) while dispatching compute passes for the heavy lifting (warping).

5.2 The CUDA Ecosystem (C++/Python)

The traditional route involves C++ with CUDA, often wrapped in PyTorch.
Throughput: CUDA allows access to hardware-specific intrinsics like warp shuffles, which can perform reductions (summing variance) without memory access. This offers the theoretical highest throughput.34
Library Support: NVIDIA provides primitives for sorting and scanning (thrust, cub) which are essential for some advanced CMAX variants.
Drawbacks:
Safety: CUDA kernels are prone to race conditions and out-of-bounds access.
Latency: Launching a CUDA kernel has a non-zero overhead ($\sim 5-20 \mu s$). For a control loop requiring $<1 ms$ latency (e.g., drone stabilization), this overhead is significant.
Hardware Lock-in: CUDA runs only on NVIDIA GPUs. Rust/wgpu runs on Vulkan, Metal, DX12, making it deployable on a MacBook, a Raspberry Pi (with Vulkan driver), or a Jetson.

5.3 Comparative Summary Table

Feature
Rust (CPU / Rayon)
Rust (wgpu)
C++ (CUDA)
Best For
Low Latency, Embedded, 1D/2D search
Portable GPU acceleration, Visualization
Massive Batch Throughput, Research
Safety
Memory Safe (Compiler enforced)
High (Validation Layers)
Low (Pointer Arithmetic)
Accumulation
Fast (Thread-local caching)
Slow (Global Atomics) unless optimized
Very Fast (Warp Intrinsics)
Deployment
Single Binary
Single Binary
Driver Dependency Hell
Hotspot Handling
Excellent (L1/L2 cache handles it)
Requires Shared Mem optimization
Requires Shared Mem optimization


6. Algorithmic Detail: Implementing "Geometry-Instructed" RPM Estimation

We present a synthesized algorithm that combines the best practices from EventPro and EV-Tach, suitable for implementation in Rust.

6.1 Step 1: Ingestion and Filtering

Input: Stream of events $\mathcal{E}$.
Action: Apply a Refractory Period Filter. If a pixel fires events too rapidly (faster than the physical dynamics of the blade allow), it is likely sensor noise or "hot pixel" artifact.
Code Concept (Rust):
Rust
// using davis-edi-rs style iterator
let filtered_events: Vec<_> = raw_events.iter()
   .filter(|e| e.timestamp - last_timestamp[e.y][e.x] > REFRACTORY_TIME)
   .collect();



6.2 Step 2: Robust Centroid Estimation

Before estimating $\omega$, we must find $\mathbf{c}$.
Method: Accumulate events over a longer window ($50ms$). Threshold the image to binary. Compute the geometric centroid of the active pixels.
Why: For a spinning fan, the time-integrated image is a disk or annulus. Its center of mass is the center of rotation. This is robust and $O(N)$.18

6.3 Step 3: Polar Transformation (The "Efficiency Hack")

Standard CMAX warps in Cartesian coordinates $(x, y)$. This requires sin/cos for every event, every iteration.
Optimization: Transform all events to Polar coordinates $(r, \theta_{ang})$ relative to $\mathbf{c}$ once.

$$r_k = \sqrt{(x_k-c_x)^2 + (y_k-c_y)^2}$$
$$\theta_{ang, k} = \text{atan2}(y_k-c_y, x_k-c_x)$$
The Linear Warp: In Polar space, rotation is a linear translation:

$$\theta'_{ang, k} = \theta_{ang, k} - \omega (t_{ref} - t_k)$$
Benefit: The warp now involves only subtraction and multiplication. No trigonometry inside the optimization loop. This speeds up the cost function evaluation by $\sim 10\times$, enabling real-time CPU performance.5

6.4 Step 4: 1D Optimization Strategy

We search for $\omega$. Since the search space is 1D (scalar RPM):
Golden Section Search: This derivative-free method is ideal because the Variance landscape is generally convex but noisy. It narrows the bracket $[\omega_{min}, \omega_{max}]$ iteratively.
Cost Function:
Compute $\theta'_{ang}$ for all events.
Build a 1D Histogram (or 2D Polar Image) of warped events.
Compute Variance of the histogram.
Note: Using a 1D histogram (projecting onto the angle axis) works if the blades are radial spokes. If they are curved, a 2D Polar IWE is needed.

6.5 Step 5: Post-Processing Edge Detection

Once $\omega^*$ is found:
Generate the final IWE.
Canny Edge Detector: Apply standard Canny detection to the IWE. Because the IWE is created from motion-compensated events, it has extremely high dynamic range and SNR.
Blade Counting: Perform a Hough Line Transform (or Hough Curve) on the edge map to count the blades. This provides a secondary verification: $f_{observed} = \text{RPM} \times N_{blades} / 60$.

7. Case Studies and Performance Metrics


7.1 Accuracy comparison

Baseline (FFT of Pixel Intensity): Accuracy $\sim 2-5\%$. Fails when lighting changes or blades are textureless.36
EV-Tach (CMAX + Clustering): Accuracy $\sim 0.03\%$ (relative error). Comparable to industrial laser tachometers.
EventPro (Geometry-Instructed): Accuracy $\sim 0.23\%$ in dynamic drone scenarios (moving drone). The slight drop compared to EV-Tach (static) is due to the complexity of ego-motion, but it remains highly effective for control loops.19

7.2 Latency and Power

Rust (CPU) Latency: On a Raspberry Pi 4, a polar-transformed CMAX loop can converge in $< 5ms$ for a batch of 10,000 events.
Power: The event camera consumes $\sim 10-20mW$. The CPU processing (Rust) consumes $\sim 1-2W$. This is significantly lower than a high-speed camera setup ($>10W$ for camera + FPGA processing).1

7.3 The "Object-Centric" Advantage

Traditional optical flow methods are "Scene-Centric"—they try to align the whole image. For RPM estimation, we are "Object-Centric." We only care about the fan.
Insight: By masking the IWE (using the EV-Tach outlier removal), we decouple the fan's rotation from the background motion. This allows measuring fan RPM even while the drone is flying and the background is rushing past.19

8. Conclusion and Future Outlook

The calculation of fast-spinning fan RPM using event cameras is a quintessential example of how computational imaging can solve problems intractable for traditional sensors. By shifting the paradigm from "capturing frames" to "integrating events," we convert the handicap of high speed into a source of high-fidelity signal.
Contrast Maximization stands as the theoretical bedrock of this approach. It transforms the temporal precision of the sensor into spatial precision in the reconstruction. The introduction of Geometry-Instructed constraints (as seen in EventPro) and Polar Domain Optimization represents the maturing of this field from generic research to application-specific engineering.
From an implementation perspective, the Rust ecosystem is rapidly becoming the gold standard for such robotic perception tasks. Its ability to provide memory safety without sacrificing the performance of SIMD and bare-metal execution offers a compelling alternative to the fragile pointer arithmetic of C++ or the latency of Python. While GPU acceleration (via wgpu or CUDA) remains necessary for dense, full-scene reconstruction, the scalar nature of RPM estimation favors the deterministic, low-latency execution of optimized CPU code.
As event sensors become ubiquitous in industrial monitoring and drone avionics, the algorithms detailed here—focusing on edge alignment, variance maximization, and geometric regularity—will form the core logic of autonomous, high-speed machine vision.

9. References and Source Attribution

CMAX Theory: 5
Rotational Warping: 1
EV-Tach Algorithm: 17
EventPro Algorithm: 19
Rust Implementation (davis-EDI-rs, aedat-rs): 26
GPU/wgpu Implementation: 32
Linear Algebra (nalgebra): 30
Objective Functions (Variance vs Entropy): 13
Works cited
Globally-Optimal Contrast Maximisation for Event Cameras - arXiv, accessed November 23, 2025, https://arxiv.org/pdf/2206.05127
EVPropNet: Detecting Drones By Finding Propellers For Mid-Air Landing And Following - Robotics, accessed November 23, 2025, https://www.roboticsproceedings.org/rss17/p074.pdf
Event-based feature tracking in a visual inertial odometry framework - PMC, accessed November 23, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC9971716/
[2206.05127] Globally-Optimal Contrast Maximisation for Event Cameras - arXiv, accessed November 23, 2025, https://arxiv.org/abs/2206.05127
A Unifying Contrast Maximization Framework for Event Cameras (CVPR'18) - YouTube, accessed November 23, 2025, https://www.youtube.com/watch?v=KFMZFhi-9Aw
Accurate Angular Velocity Estimation with an Event Camera - Robotics and Perception Group, accessed November 23, 2025, https://rpg.ifi.uzh.ch/docs/RAL16_Gallego.pdf
CMax-SLAM: Event-based Rotational-Motion Bundle Adjustment and SLAM System using Contrast Maximization - arXiv, accessed November 23, 2025, https://arxiv.org/html/2403.08119v1
Globally Optimal Contrast Maximisation for Event-Based Motion Estimation - CVF Open Access, accessed November 23, 2025, https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Globally_Optimal_Contrast_Maximisation_for_Event-Based_Motion_Estimation_CVPR_2020_paper.pdf
Visual Odometry with an Event Camera Using Continuous Ray Warping and Volumetric Contrast Maximization - NIH, accessed November 23, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC9370870/
[1804.01306] A Unifying Contrast Maximization Framework for Event Cameras, with Applications to Motion, Depth, and Optical Flow Estimation - arXiv, accessed November 23, 2025, https://arxiv.org/abs/1804.01306
Event Collapse in Contrast Maximization Frameworks - arXiv, accessed November 23, 2025, https://arxiv.org/pdf/2207.04007
Focus Is All You Need: Loss Functions for Event-Based Vision - CVF Open Access, accessed November 23, 2025, https://openaccess.thecvf.com/content_CVPR_2019/papers/Gallego_Focus_Is_All_You_Need_Loss_Functions_for_Event-Based_Vision_CVPR_2019_paper.pdf
Event Collapse in Contrast Maximization Frameworks - MDPI, accessed November 23, 2025, https://www.mdpi.com/1424-8220/22/14/5190
Secrets of Edge-Informed Contrast Maximization for Event-Based Vision - CVF Open Access, accessed November 23, 2025, https://openaccess.thecvf.com/content/WACV2025/papers/Karmokar_Secrets_of_Edge-Informed_Contrast_Maximization_for_Event-Based_Vision_WACV_2025_paper.pdf
Accuracy and Speed Improvement of Event Camera Motion Estimation Using a Bird's-Eye View Transformation - ResearchGate, accessed November 23, 2025, https://www.researchgate.net/publication/357995857_Accuracy_and_Speed_Improvement_of_Event_Camera_Motion_Estimation_Using_a_Bird's-Eye_View_Transformation
Recursive Contrast Maximization for Event-Based High-Frequency Motion Estimation - IEEE Xplore, accessed November 23, 2025, https://ieeexplore.ieee.org/iel7/6287639/9668973/09966595.pdf
High Speed Rotation Estimation with Dynamic Vision Sensors - arXiv, accessed November 23, 2025, https://arxiv.org/pdf/2209.02205
EV-Tach: A Handheld Rotational Speed Estimation System With Event Camera, accessed November 23, 2025, https://www.semanticscholar.org/paper/EV-Tach%3A-A-Handheld-Rotational-Speed-Estimation-Zhao-Shen/e6f8af5c47a66feaefab511e4cc3b054f074d3c0
Count Every Rotation and Every Rotation Counts: Exploring Drone Dynamics via Propeller Sensing - arXiv, accessed November 23, 2025, https://arxiv.org/html/2511.13100v1
EE3P: Event-based Estimation of Periodic Phenomena Properties - arXiv, accessed November 23, 2025, https://arxiv.org/html/2402.14958v1
EV-Tach: A Handheld Rotational Speed Estimation System With Event Camera, accessed November 23, 2025, http://ieeexplore.ieee.org/document/10337756/
EV-Tach: A Handheld Rotational Speed Estimation System With Event Camera | Request PDF - ResearchGate, accessed November 23, 2025, https://www.researchgate.net/publication/376165946_EV-Tach_A_Handheld_Rotational_Speed_Estimation_System_With_Event_Camera
[Papierüberprüfung] Count Every Rotation and Every Rotation Counts: Exploring Drone Dynamics via Propeller Sensing - Moonlight, accessed November 23, 2025, https://www.themoonlight.io/de/review/count-every-rotation-and-every-rotation-counts-exploring-drone-dynamics-via-propeller-sensing
[Literature Review] Count Every Rotation and Every Rotation Counts: Exploring Drone Dynamics via Propeller Sensing - Moonlight, accessed November 23, 2025, https://www.themoonlight.io/en/review/count-every-rotation-and-every-rotation-counts-exploring-drone-dynamics-via-propeller-sensing
[Revue de papier] Count Every Rotation and Every Rotation Counts: Exploring Drone Dynamics via Propeller Sensing - Moonlight, accessed November 23, 2025, https://www.themoonlight.io/fr/review/count-every-rotation-and-every-rotation-counts-exploring-drone-dynamics-via-propeller-sensing
ac-freeman/davis-EDI-rs - GitHub, accessed November 23, 2025, https://github.com/ac-freeman/davis-EDI-rs
Science — list of Rust libraries/crates // Lib.rs, accessed November 23, 2025, https://lib.rs/science
uzh-rpg/event-based_vision_resources: Event-based Vision Resources. Community effort to collect knowledge on event-based vision technology (papers, workshops, datasets, code, videos, etc) - GitHub, accessed November 23, 2025, https://github.com/uzh-rpg/event-based_vision_resources
ac-freeman/adder-codec-rs: A unified framework for event-based video. Encoder/transcoder/decoder for ADΔER (Address, Decimation, Δt Event Representation) video streams. - GitHub, accessed November 23, 2025, https://github.com/ac-freeman/adder-codec-rs
Rotation2 in nalgebra::geometry - Rust - Docs.rs, accessed November 23, 2025, https://docs.rs/nalgebra/latest/nalgebra/geometry/type.Rotation2.html
Introduction to nalgebra | Rapier, accessed November 23, 2025, https://rapier.rs/docs/user_guides/rust/introduction_to_nalgebra/
A Better Camera | Learn Wgpu, accessed November 23, 2025, https://sotrh.github.io/learn-wgpu/intermediate/tutorial12-camera/
Exploiting GPU Caches from the Browser with WebGPU - Graz University of Technology, accessed November 23, 2025, https://tugraz.elsevierpure.com/ws/portalfiles/portal/93578583/88958.pdf
GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell | NVIDIA Technical Blog, accessed November 23, 2025, https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
Parallel Radix sort confusions & options : r/GraphicsProgramming - Reddit, accessed November 23, 2025, https://www.reddit.com/r/GraphicsProgramming/comments/14c7gok/parallel_radix_sort_confusions_options/
EEPPR: event-based estimation of periodic phenomena rate using correlation in 3D, accessed November 23, 2025, https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13517/135170S/EEPPR--event-based-estimation-of-periodic-phenomena-rate-using/10.1117/12.3055033.full
EE3P3D: Event-based Estimation of Periodic Phenomena Frequency using 3D Correlation, accessed November 23, 2025, https://arxiv.org/html/2408.06899v1
Neuromorphic Event-based Sensing and Computing - PeAR WPI, accessed November 23, 2025, https://pear.wpi.edu/eventvision.html
EF-Calib: Spatiotemporal Calibration of Event- and Frame-Based Cameras Using Continuous-Time Trajectories | Request PDF - ResearchGate, accessed November 23, 2025, https://www.researchgate.net/publication/384620487_EF-Calib_Spatiotemporal_Calibration_of_Event-_and_Frame-Based_Cameras_Using_Continuous-Time_Trajectories
adder-viz: Real-Time Visualization Software for Transcoding Event Video - arXiv, accessed November 23, 2025, https://arxiv.org/html/2508.14996v2
WebGPU Compute Shaders - Image Histogram, accessed November 23, 2025, https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders-histogram.html
WebGPU Compute Shaders - Image Histogram Part 2, accessed November 23, 2025, https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders-histogram-part-2.html
Wumpf/blub: 3D fluid simulation experiments in Rust, using WebGPU-rs (WIP) - GitHub, accessed November 23, 2025, https://github.com/Wumpf/blub
