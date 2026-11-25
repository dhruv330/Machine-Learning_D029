# Linear Regression Lab

An interactive visualisation of simple linear regression — `y = mx + b` — fitted
two different ways so you can watch them race each other.

No build step, no dependencies, no server. Open `index.html` in a browser.

## Files

| File | What's in it |
|---|---|
| `index.html` | Markup: the two canvases, the stats panel, the controls |
| `style.css` | Design tokens (light/dark), layout, control styling |
| `script.js` | The maths, the canvas rendering, the interaction |

## How to use it

- **Click** the plot to drop a data point
- **Drag** a point to move it — drag one far away to see how a single outlier drags the whole line
- **Shift-click** a point to delete it
- **Train** runs gradient descent; **Step** advances exactly one iteration
- **Learning rate** slider — push it past ~0.1 and the run explodes on purpose

## The two fits

**Green solid line — gradient descent.** Starts from a random guess and walks
downhill. This is the one that animates, and it's how real models with no
closed-form solution are actually trained.

**Magenta dashed line — ordinary least squares.** The exact answer, computed
directly in one pass. It's the target gradient descent is crawling toward, which
is why it's drawn as a fixed reference.

## The maths

**Cost — mean squared error.** This is the number being minimised. Each point's
vertical distance to the line (its *residual*, drawn in yellow) is squared, and
the squares are averaged. Squaring is what makes big misses hurt disproportionately.

```
MSE = (1/n) · Σ (yᵢ − (m·xᵢ + b))²
```

**Closed form (normal equation).** For a straight line, calculus gives the exact
minimum directly — no iteration, no learning rate, no tuning:

```
m = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
b = (Σy − m·Σx) / n
```

**Gradient descent.** Take the partial derivatives of MSE with respect to `m` and
`b`, then repeatedly step against the gradient:

```
∂MSE/∂m = (−2/n) · Σ xᵢ(yᵢ − ŷᵢ)
∂MSE/∂b = (−2/n) · Σ (yᵢ − ŷᵢ)

m ← m − lr · ∂MSE/∂m
b ← b − lr · ∂MSE/∂b
```

**R² (coefficient of determination).** How much better the line is than just
predicting the mean of `y` every time. 1.0 is perfect; 0 means the line is no
better than the mean; **negative is entirely possible** — you'll see it before
training, when the random starting line is worse than a flat guess.

```
R² = 1 − SSres/SStot
```

## Things worth trying

- **Crank the learning rate to ~0.5.** The line diverges in about four iterations
  and the app tells you why. Too-large steps overshoot the minimum and each
  correction overshoots further — the exact reason learning rate matters.
- **Set noise to 0.** R² snaps to 1.000 and the residuals vanish.
- **Drag one point to a far corner.** The line swings noticeably. That's least
  squares' outlier sensitivity, live.
- **Press Step repeatedly at a low learning rate.** You can watch the line inch
  toward the dashed target one gradient at a time.
- **Watch the "best possible" line in the cost chart.** The gap between the green
  descent curve and that dashed floor is exactly how much accuracy is still on
  the table.

## Notes on the visuals

The four chart colours (blue data / green descent / magenta OLS / yellow
residuals) were validated for colourblind separation and contrast against both
the light and dark surfaces, in all pairs. The slot ordering is the safety
mechanism, so don't reshuffle them casually — a plausible-looking swap like
"orange line, red residuals" fails the gates badly (orange↔red is nearly
indistinguishable even with full colour vision). Dashed-vs-solid carries the
OLS/GD distinction as a second channel, so the pair never relies on hue alone.
