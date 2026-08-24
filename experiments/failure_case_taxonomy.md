# Failure-Case Taxonomy

Systematic documentation of model failure modes. For each category, collect
representative examples and count occurrences. Only include figures where
dataset/publication licensing permits.

## Instructions

1. Run the classifier on your test set (or a held-out subset)
2. Collect frames where the prediction is **incorrect**
3. Categorize each failure using the taxonomy below
4. Count occurrences per category
5. Select 1–3 representative images per category for the paper
6. Check dataset license before including any images

---

## Taxonomy Categories

### 1. Eyewear-Related
| Sub-category | Count | Example Frame IDs | Notes |
|--------------|-------|-------------------|-------|
| Sunglasses | | | Opaque lenses block eye cues |
| Clear glasses + glare | | | Reflection obscures eyelids |
| Large frames | | | Frame edges near eye boundary |

### 2. Illumination
| Sub-category | Count | Example Frame IDs | Notes |
|--------------|-------|-------------------|-------|
| Very low light | | | Under ~20 lux |
| Strong backlight | | | Window behind subject |
| Direct glare | | | Headlight or overhead lamp |
| Mixed/shadow | | | Half-face shadow |

### 3. Head Pose / Orientation
| Sub-category | Count | Example Frame IDs | Notes |
|--------------|-------|-------------------|-------|
| Extreme lateral turn | | | >60° from camera |
| Head tilted down | | | Chin near chest |
| Head tilted up | | | Looking at mirror/roof |

### 4. Occlusion
| Sub-category | Count | Example Frame IDs | Notes |
|--------------|-------|-------------------|-------|
| Hand near face | | | Rubbing eyes, yawning |
| Phone in front of face | | | Talking on phone |
| Mask / face covering | | | Surgical or cloth mask |
| Hair covering face | | | Long hair, wind |

### 5. Transient Expressions (Not Fatigue)
| Sub-category | Count | Example Frame IDs | Notes |
|--------------|-------|-------------------|-------|
| Normal blink | | | Brief eye closure <0.4s |
| Yawn (not drowsy) | | | Conscious yawning |
| Laughing / talking | | | Wide mouth, raised cheeks |
| Focused squint | | | Concentration, not fatigue |

### 6. Camera / Motion Artifacts
| Sub-category | Count | Example Frame IDs | Notes |
|--------------|-------|-------------------|-------|
| Motion blur | | | Fast head movement |
| Out-of-focus | | | Camera autofocus lag |
| Very close to camera | | | Face fills >80% of frame |
| Very far from camera | | | Face <15% of frame |

### 7. Transitional States
| Sub-category | Count | Example Frame IDs | Notes |
|--------------|-------|-------------------|-------|
| Falling asleep | | | Gradual transition zone |
| Just woke up | | | Groggy but not classified |
| Intermittent drowsiness | | | Alternating alert/drowsy |

---

## Summary Table (fill after analysis)

| Category | Total Failures | % of All Failures | Severity |
|----------|---------------|-------------------|----------|
| Eyewear | | | |
| Illumination | | | |
| Head Pose | | | |
| Occlusion | | | |
| Transient Expressions | | | |
| Camera / Motion | | | |
| Transitional States | | | |
| **Total** | | **100%** | |

## Licensing Note

Before including any example images in the paper, verify:
- [ ] Dataset license permits redistribution of individual frames
- [ ] Subject consent covers publication (if applicable)
- [ ] Images are de-identified (no visible names/tags)
- [ ] Supplementary material license allows figure inclusion
