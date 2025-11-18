package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

// --- Constants ---
const (
	WIDTH     = 120
	HEIGHT    = 40
	PARTICLES = 280
	LAYERS    = 3
)

// --- Structs ---

type Vec2 struct {
	x, y float64
}

func (v Vec2) Add(o Vec2) Vec2 { return Vec2{v.x + o.x, v.y + o.y} }
func (v Vec2) Sub(o Vec2) Vec2 { return Vec2{v.x - o.x, v.y - o.y} }
func (v Vec2) Mul(s float64) Vec2 { return Vec2{v.x * s, v.y * s} }
func (v Vec2) Len() float64 { return math.Sqrt(v.x*v.x + v.y*v.y) }
func (v Vec2) Norm() Vec2 {
	l := v.Len()
	if l == 0 {
		return Vec2{0, 0}
	}
	return Vec2{v.x / l, v.y / l}
}
func (v Vec2) Rot(a float64) Vec2 {
	c, s := math.Cos(a), math.Sin(a)
	return Vec2{v.x*c - v.y*s, v.x*s + v.y*c}
}

type Particle struct {
	pos, vel, acc            Vec2
	hue, life, age, layer, seed float64
}

type Cell struct {
	ch    rune
	color int
}

// --- State ---

var (
	particles    [PARTICLES]Particle
	grid         [HEIGHT][WIDTH]Cell
	globalTime   float64
	modeTime     float64
	modeIndex    int
	frameCount   int
	paletteShift float64
)

// --- Math Helpers ---

func frand(min, max float64) float64 {
	return min + (max-min)*rand.Float64()
}

func clamp(v, a, b float64) float64 {
	if v < a {
		return a
	}
	if v > b {
		return b
	}
	return v
}

func lerp(a, b, t float64) float64 {
	return a + (b-a)*t
}

func smoothstep(edge0, edge1, x float64) float64 {
	if edge0 == edge1 {
		return 0.0
	}
	t := clamp((x-edge0)/(edge1-edge0), 0.0, 1.0)
	return t * t * (3.0 - 2.0*t)
}

func hashDouble(x, y, seed float64) float64 {
	h := x*37.0 + y*17.0 + seed*13.0
	s := math.Sin(h*12.9898) * 43758.5453
	return s - math.Floor(s)
}

func noise2d(x, y, seed float64) float64 {
	xi := math.Floor(x)
	yi := math.Floor(y)
	xf := x - xi
	yf := y - yi
	h00 := hashDouble(xi+0.0, yi+0.0, seed)
	h10 := hashDouble(xi+1.0, yi+0.0, seed)
	h01 := hashDouble(xi+0.0, yi+1.0, seed)
	h11 := hashDouble(xi+1.0, yi+1.0, seed)
	ux := xf * xf * (3.0 - 2.0*xf)
	uy := yf * yf * (3.0 - 2.0*yf)
	a := h00 + (h10-h00)*ux
	b := h01 + (h11-h01)*ux
	return a + (b-a)*uy
}

// --- Graphics/Grid Helpers ---

func clearGrid() {
	for y := 0; y < HEIGHT; y++ {
		for x := 0; x < WIDTH; x++ {
			grid[y][x].ch = ' '
			grid[y][x].color = 0
		}
	}
}

func putCell(x, y int, ch rune, color int) {
	if x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT {
		return
	}
	if color >= grid[y][x].color {
		grid[y][x].ch = ch
		grid[y][x].color = color
	}
}

func hsvToRgb(h, s, v float64) (int, int, int) {
	c := v * s
	hh := h / 60.0
	x := c * (1.0 - math.Abs(math.Mod(hh, 2.0)-1.0))
	m := v - c
	var rr, gg, bb float64

	if hh < 0.0 {
		hh = 0.0 // Should not happen with valid inputs
	}

	switch {
	case hh < 1.0:
		rr, gg, bb = c, x, 0.0
	case hh < 2.0:
		rr, gg, bb = x, c, 0.0
	case hh < 3.0:
		rr, gg, bb = 0.0, c, x
	case hh < 4.0:
		rr, gg, bb = 0.0, x, c
	case hh < 5.0:
		rr, gg, bb = x, 0.0, c
	default:
		rr, gg, bb = c, 0.0, x
	}

	return int((rr + m) * 255.0), int((gg + m) * 255.0), int((bb + m) * 255.0)
}

func rgbToAnsi256(r, g, b int) int {
	ri, gi, bi := r/51, g/51, b/51
	return 16 + 36*ri + 6*gi + bi
}

func hueToAnsi(hue, layer, alpha float64) int {
	h := math.Mod(hue+paletteShift, 360.0)
	if h < 0 {
		h += 360.0
	}
	s := clamp(0.75+0.2*layer, 0.0, 1.0)
	v := clamp(0.35+0.6*alpha, 0.0, 1.0)
	r, g, b := hsvToRgb(h, s, v)
	return rgbToAnsi256(r, g, b)
}

func flushGrid(w *bufio.Writer) {
	w.WriteString("\x1b[H")
	lastColor := -1
	for y := 0; y < HEIGHT; y++ {
		for x := 0; x < WIDTH; x++ {
			c := grid[y][x]
			if c.color <= 0 {
				if lastColor != 0 {
					w.WriteString("\x1b[0m")
					lastColor = 0
				}
				w.WriteRune(' ')
			} else {
				if c.color != lastColor {
					fmt.Fprintf(w, "\x1b[38;5;%dm", c.color)
					lastColor = c.color
				}
				w.WriteRune(c.ch)
			}
		}
		if lastColor != 0 {
			w.WriteString("\x1b[0m")
			lastColor = 0
		}
		w.WriteRune('\n')
	}
	w.WriteString("\x1b[0m")
	w.Flush()
}

// --- Simulation Logic ---

func initParticle(idx int) {
	cx, cy := float64(WIDTH)/2.0, float64(HEIGHT)/2.0
	r := frand(0.0, float64(HEIGHT)*0.45)
	a := frand(0.0, 2.0*math.Pi)
	layer := float64(idx%LAYERS) / float64(LAYERS)
	life := frand(4.0, 18.0)

	particles[idx] = Particle{
		pos:   Vec2{cx + math.Cos(a)*r, cy + math.Sin(a)*r*0.55},
		vel:   Vec2{0, 0},
		acc:   Vec2{0, 0},
		hue:   frand(0.0, 360.0),
		life:  life,
		age:   frand(0.0, life*0.2),
		layer: layer,
		seed:  frand(0.0, 1000.0),
	}
}

func initParticles() {
	for i := 0; i < PARTICLES; i++ {
		initParticle(i)
	}
}

func updateParticle(p *Particle, dt, t float64) {
	p.acc = Vec2{0, 0}
	pos := p.pos

	// Apply Field
	cx, cy := float64(WIDTH)/2.0, float64(HEIGHT)/2.0
	d := pos.Sub(Vec2{cx, cy})
	r := d.Len() + 0.001
	
	var ang, strength, factor float64

	switch modeIndex {
	case 0: // Spiral
		swirl := math.Sin(r*0.12 + t*0.25)
		ang = math.Atan2(d.y, d.x) + 0.9 + swirl*0.8 + math.Sin(t*0.05)*0.4
		wave := math.Sin(r*0.2-t*0.5)*0.5 + 0.5
		strength = wave * (1.0 / (1.0 + r*0.03)) * (0.7 + 0.4*math.Sin(t*0.7))
		factor = 1.4
	case 1: // Rings
		waves := math.Sin(r*0.18-t*0.65) + math.Sin(r*0.07+t*0.3)
		ang = math.Atan2(d.y, d.x) + math.Sin(t*0.17)*0.6 + waves*0.15
		wave := math.Sin(r*0.2-t*0.5)*0.5 + 0.5
		strength = wave * (1.0 / (1.0 + r*0.03)) * (0.7 + 0.4*math.Sin(t*0.7)) * 1.2
		factor = 1.0
	default: // Lens
		lens := smoothstep(0.0, float64(WIDTH)*0.35, r)
		lens2 := 1.0 - smoothstep(float64(WIDTH)*0.12, float64(WIDTH)*0.55, r)
		pulse := math.Sin(t*0.4) * 0.4
		swirl := math.Sin(r*0.19 + t*0.7)
		ang = math.Atan2(d.y, d.x) + lens*1.8 + lens2*(-1.2) + pulse + swirl*0.3
		wave := math.Sin(r*0.2-t*0.5)*0.5 + 0.5
		strength = wave * (1.0 / (1.0 + r*0.03)) * (0.7 + 0.4*math.Sin(t*0.7)) * 1.4
		factor = 1.3
	}

	dir := Vec2{math.Cos(ang), math.Sin(ang)}
	force := dir.Mul(strength * (0.4 + p.layer*0.9) * factor)
	
	// Center attraction
	towardsCenter := Vec2{cx, cy}.Sub(pos)
	rc := towardsCenter.Len() + 0.001
	towardsCenter = towardsCenter.Norm().Mul(0.05 / rc)

	// Jitter
	n := noise2d(pos.x*0.1, pos.y*0.1, p.seed+t*0.15)
	m := noise2d(pos.x*0.05+10.0, pos.y*0.05-7.0, p.seed+t*0.2)
	jitter := Vec2{math.Cos(n*6.28) * 0.1, math.Sin(m*6.28) * 0.1}

	p.acc = p.acc.Add(force).Add(towardsCenter).Add(jitter)

	// Apply Orbit
	offset := math.Sin(t*0.13 + p.layer*5.0)
	orbitR := lerp(float64(HEIGHT)*0.12, float64(HEIGHT)*0.4, (math.Sin(t*0.05+p.layer*3.0)+1.0)*0.5)
	angle := globalTime*(0.3+p.layer*0.7) + p.seed
	ringCenter := Vec2{cx + math.Cos(angle)*orbitR*0.75, cy + math.Sin(angle*1.1)*orbitR*0.4}
	d = ringCenter.Sub(p.pos)
	dist := d.Len() + 0.001
	orbitDir := d.Norm().Mul(0.25 + 0.6*(1.0/(1.0+dist*0.2)))
	tangent := orbitDir.Rot(offset * 0.7)
	p.acc = p.acc.Add(tangent)

	// Apply Noise Orbit
	scale := 0.12 + 0.05*p.layer
	nx, ny := p.pos.x*scale, p.pos.y*scale
	na := noise2d(nx, ny, p.seed+t*0.17) * 6.28318
	nb := noise2d(nx+17.0, ny+9.0, p.seed-t*0.21) * 6.28318
	f1 := Vec2{math.Cos(na), math.Sin(na)}
	f2 := Vec2{math.Cos(nb), math.Sin(nb)}
	p.acc = p.acc.Add(f1.Mul(0.3).Add(f2.Mul(0.25)))

	// Integrate
	p.vel = p.vel.Add(p.acc.Mul(dt * 0.7))
	maxSpeed := 2.0 + p.layer*3.0
	if p.vel.Len() > maxSpeed {
		p.vel = p.vel.Norm().Mul(maxSpeed)
	}
	p.pos = p.pos.Add(p.vel.Mul(dt))

	// Bounds check
	if p.pos.x < -10.0 || p.pos.x > WIDTH+10.0 || p.pos.y < -10.0 || p.pos.y > HEIGHT+10.0 {
		// Re-init specific index by finding ptr offset? 
		// Actually simpler to just re-init "this" particle logic in place
		// but we need the index for the layer logic.
		// Simplest way: we cheat and assume we know index or generate new rand layer.
		// But strictly following C, we need index. We can deduce layer approx or just pass index.
		// Let's just reset state manually without index dependance for `init`.
		// Actually, let's just re-call initParticle. We need to pass index to update function or store it.
		// We will handle this in the loop.
		p.age = p.life + 1 // Mark for reset
	}

	p.age += dt * (0.8 + p.layer*0.7)
}

func sampleChar(k, layer, flicker float64) rune {
	chars0 := []rune(" .:-=+*#%@")
	chars1 := []rune(" .,:;irsXA253hMHGS#9B&@")
	chars2 := []rune(" `'\"^\",:;Il!i><~+_-?][}{1)(|\\/")

	f := clamp(k+layer*0.4+flicker*0.3, 0.0, 1.0)
	
	if layer < 0.33 {
		idx := int(f * float64(len(chars2)-1))
		return chars2[idx]
	} else if layer < 0.66 {
		idx := int(f * float64(len(chars1)-1))
		return chars1[idx]
	} else {
		idx := int(f * float64(len(chars0)-1))
		return chars0[idx]
	}
}

func drawParticle(p *Particle, t float64) {
	fadeIn := smoothstep(0.0, p.life*0.2, p.age)
	fadeOut := 1.0 - smoothstep(p.life*0.3, p.life, p.age)
	energy := fadeIn * fadeOut
	glow := smoothstep(0.0, 1.0, energy)
	flicker := math.Sin(t*6.0+p.seed*4.0)*0.5 + 0.5
	alpha := clamp(glow*(0.3+flicker*0.9), 0.0, 1.0)

	gx, gy := int(math.Round(p.pos.x)), int(math.Round(p.pos.y))
	color := hueToAnsi(p.hue+t*(4.0+p.layer*20.0), p.layer, alpha)
	ch := sampleChar(alpha, p.layer, flicker)
	
	putCell(gx, gy, ch, color)

	// Glow
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			falloff := 1.0 / (1.0 + float64(dx*dx+dy*dy))
			aa := alpha * falloff * 0.7
			cc := hueToAnsi(p.hue+t*2.0, p.layer*0.6, aa)
			ch2 := sampleChar(aa*0.8, p.layer*0.7, flicker*0.4)
			if aa > 0.05 {
				putCell(gx+dx, gy+dy, ch2, cc)
			}
		}
	}
}

func drawTrails(t float64) {
	for i := 0; i < PARTICLES; i++ {
		p := &particles[i]
		trailK := 1.0 - smoothstep(0.0, p.life*0.4, p.age)
		steps := 3
		for s := 1; s <= steps; s++ {
			u := float64(s) / float64(steps+1)
			pos := p.pos.Sub(p.vel.Mul(u * 0.9))
			alpha := trailK * (1.0 - u*0.9)
			gx, gy := int(math.Round(pos.x)), int(math.Round(pos.y))
			color := hueToAnsi(p.hue+t*(1.0+p.layer*5.0), p.layer*0.6, alpha)
			ch := sampleChar(alpha*0.8, p.layer*0.6, u)
			putCell(gx, gy, ch, color)
		}
	}
}

func drawCore(t float64) {
	cx, cy := float64(WIDTH)/2.0, float64(HEIGHT)/2.0
	r := float64(HEIGHT)*0.12 + math.Sin(t*0.6)*float64(HEIGHT)*0.025
	steps := int(r * 8.0)
	for i := 0; i < steps; i++ {
		a := float64(i) / float64(steps) * 2.0 * math.Pi
		rr := r * (0.8 + 0.4*math.Sin(t*0.9+float64(i)*0.3))
		x := cx + math.Cos(a)*rr
		y := cy + math.Sin(a)*rr*0.7
		k := smoothstep(0.0, float64(HEIGHT)*0.2, rr)
		alpha := 1.0 - k
		color := hueToAnsi(180.0+math.Sin(t*0.4)*80.0, 0.8, alpha)
		ch := sampleChar(alpha, 0.8, 0.5)
		putCell(int(math.Round(x)), int(math.Round(y)), ch, color)
	}
}

func drawRings(t float64) {
	cx, cy := float64(WIDTH)/2.0, float64(HEIGHT)/2.0
	for ring := 0; ring < 3; ring++ {
		baseR := float64(HEIGHT) * (0.12 + 0.14*float64(ring))
		wobble := math.Sin(t*(0.4+float64(ring)*0.13)+float64(ring)*2.0) * float64(HEIGHT) * 0.03
		r := baseR + wobble
		steps := int(r * 7.0)
		for i := 0; i < steps; i++ {
			a := float64(i) / float64(steps) * 2.0 * math.Pi
			x := cx + math.Cos(a)*r*(1.04+float64(ring)*0.05)
			y := cy + math.Sin(a)*r*(0.7+float64(ring)*0.08)
			k := smoothstep(float64(HEIGHT)*0.05, float64(HEIGHT)*0.55, r)
			alpha := 0.35 + (1.0-k)*0.5
			color := hueToAnsi(40.0+float64(ring)*90.0+t*3.0, 0.4+float64(ring)*0.3, alpha)
			ch := sampleChar(alpha, 0.4+float64(ring)*0.3, 0.5)
			putCell(int(math.Round(x)), int(math.Round(y)), ch, color)
		}
	}
}

func drawWaveText(t float64) {
	text := "emergent orbit"
	txtLen := len(text)
	baseY := float64(HEIGHT - 4)
	span := float64(WIDTH) * 0.66
	startX := (float64(WIDTH) - span) * 0.5
	
	for i, r := range text {
		u := float64(i) / float64(txtLen-1)
		x := startX + u*span
		phase := t*0.7 + u*6.0
		dy := math.Sin(phase)*2.0 + math.Sin(phase*0.5)*1.5
		y := baseY + dy
		w := (math.Sin(phase*1.7) + 1.0) * 0.5
		alpha := 0.4 + w*0.6
		color := hueToAnsi(300.0+w*120.0, 0.9, alpha)
		putCell(int(math.Round(x)), int(math.Round(y)), r, color)
		putCell(int(math.Round(x)), int(math.Round(y)-1), '.', color)
	}
}

func drawModeIndicator(t float64) {
	names := []string{"spiral field", "ring waves", "lens flow"}
	label := names[modeIndex%3]
	for i, r := range label {
		k := float64(i) / float64(len(label)-1)
		alpha := 0.6 + 0.4*math.Sin(t*2.0+k*3.0)
		color := hueToAnsi(200.0+k*80.0, 0.3, alpha)
		putCell(2+i, 1, r, color)
	}
	
	fpsStr := fmt.Sprintf("frame %06d", frameCount)
	cx := WIDTH - 18
	for i, r := range fpsStr {
		color := hueToAnsi(120.0+float64(i)*10.0, 0.5, 0.7)
		putCell(cx+i, 1, r, color)
	}
}

func maybeChangeMode(dt float64) {
	modeTime += dt
	if modeTime > 26.0 {
		modeTime = 0.0
		modeIndex = (modeIndex + 1) % 3
		initParticles()
	}
}

// --- Main ---

func main() {
	rand.Seed(time.Now().UnixNano())
	
	writer := bufio.NewWriter(os.Stdout)
	writer.WriteString("\x1b[2J\x1b[H")
	writer.Flush()

	initParticles()
	lastTime := time.Now()

	for {
		currTime := time.Now()
		dt := currTime.Sub(lastTime).Seconds()
		if dt <= 0 { dt = 0.016 }
		if dt > 0.1 { dt = 0.1 }
		lastTime = currTime

		globalTime += dt
		paletteShift = math.Sin(globalTime*0.21) * 80.0

		maybeChangeMode(dt)
		clearGrid()

		for i := 0; i < PARTICLES; i++ {
			// Scale dt by 7.0 to match the C implementation speed
			updateParticle(&particles[i], dt*7.0, globalTime)
			if particles[i].age > particles[i].life {
				initParticle(i)
			}
		}

		drawTrails(globalTime)
		drawCore(globalTime)
		drawRings(globalTime)
		drawWaveText(globalTime)
		drawModeIndicator(globalTime)

		for i := 0; i < PARTICLES; i++ {
			drawParticle(&particles[i], globalTime)
		}

		flushGrid(writer)
		frameCount++

		// Reset periodically to prevent float precision issues over very long runs
		if frameCount > 0 && frameCount%9000 == 0 {
			initParticles()
		}

		time.Sleep(33 * time.Millisecond)
	}
}
