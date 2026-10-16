// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026 lvd
//
// omvcon view: a script's frames in a browser page, and the page's clicks
// and boxes back to the script.
//
// Theory of operation. view runs the script through the raw REPL as run
// does, after one exec that installs omvview.py (embedded in this binary)
// as the module omvview with ACTIVE set; see that file for the board side.
// The script's output is then one stream holding text and frame packets:
//
//	ff 4f 4d 56 46   magic, b"\xffOMVF"
//	u32 le           JPEG length n
//	u32 le           crc32 (IEEE, as zlib and binascii) of the JPEG
//	n bytes          the JPEG
//
// Text never holds 0xFF, so the demultiplexer below copies bytes to the
// terminal until it sees 0xFF (a packet, or a stray byte that is printed
// and skipped) or 0x04, which ends the script's output in the raw REPL.
// A packet is consumed by its length, so the 0x04 bytes inside JPEGs are
// never taken for that end. Packets with a bad crc are counted and dropped.
//
// The HTTP side serves the newest frame: /stream as MJPEG
// (multipart/x-mixed-replace), /frame.jpg for one, /status as JSON, and
// the page itself. The page posts "select", "lock", "unlock", "click X Y", or
// "box X Y W H" in JPEG pixels to /event; the handler checks and reformats
// the line, so only digits, letters, spaces and a newline reach the board
// (0x03 or 0x04 there would interrupt the script or end the raw REPL input).
// The main loop writes queued events between reads, so the port has one user.

package main

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/crc32"
	"image/jpeg"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed omvview.py
var omvviewSource string

// injectCode makes omvview importable without a copy on the card. A module
// object cannot be made from Python on this MicroPython (the module type
// has no constructor), so the source runs in a dict and a class carries its
// names; import finds whatever sys.modules holds. The functions keep the
// dict as their globals, which is why setting omvview.X from the script
// changes nothing inside them.
const injectCode = `import sys
_d = {}
exec(%s, _d)
_d['ACTIVE'] = True
class omvview:
    pass
for _k in _d:
    if not _k.startswith('__'):
        setattr(omvview, _k, _d[_k])
sys.modules['omvview'] = omvview
del _d, _k, omvview
`

var (
	frameMagic = []byte("\xffOMVF")
	maxFrame   = 4 << 20 // a length beyond this is a corrupt header, not a frame
)

// debugArgv is prepended to the script so the board sees --debug. The raw
// REPL runs the file body with whatever argv the firmware already has.
const debugArgv = `import sys
try:
    sys.argv.append("--debug")
except Exception:
    sys.argv = ["--debug"]
`

func withDebug(code string, debug bool) string {
	if !debug {
		return code
	}
	return debugArgv + "\n" + code
}

// recordFPS is the Tiny1-C rate. Each accepted JPEG is one video frame.
const recordFPS = 12

// recordScale keeps the one-pixel lock box sharp on a 256x192 frame.
const recordScale = "scale=1024:768:flags=neighbor"

// recordQueue is about two seconds. A full queue drops a frame instead of
// stalling the console reader.
const recordQueue = 24

func ffmpegArgs(path string) []string {
	return []string{
		"-y",
		"-loglevel", "error",
		"-f", "image2pipe",
		"-c:v", "mjpeg",
		"-framerate", strconv.Itoa(recordFPS),
		"-i", "pipe:0",
		"-an",
		"-vf", recordScale,
		"-c:v", "libx264",
		"-pix_fmt", "yuv420p",
		"-movflags", "+faststart",
		path,
	}
}

// recorder encodes the frames the page shows. ffmpeg reads a JPEG pipe so
// the host does not keep the whole session in memory.
type recorder struct {
	path   string
	cmd    *exec.Cmd
	stderr *bytes.Buffer
	frames chan []byte
	pushed int
	drop   int
	wrote  int
	err    error
	done   chan struct{}
}

func startRecorder(path string) (*recorder, error) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, errors.New("ffmpeg is not on PATH")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	cmd := exec.Command("ffmpeg", ffmpegArgs(path)...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr
	if err := cmd.Start(); err != nil {
		stdin.Close()
		return nil, err
	}
	r := &recorder{
		path:   path,
		cmd:    cmd,
		stderr: stderr,
		frames: make(chan []byte, recordQueue),
		done:   make(chan struct{}),
	}
	go r.loop(stdin)
	fmt.Fprintf(os.Stderr, "omvcon: recording %s\n", path)
	return r, nil
}

func (r *recorder) loop(stdin io.WriteCloser) {
	defer close(r.done)
	var err error
	for j := range r.frames {
		if err != nil {
			continue
		}
		if _, werr := stdin.Write(j); werr != nil {
			err = werr
			continue
		}
		r.wrote++
	}
	stdin.Close()
	if waitErr := r.cmd.Wait(); err == nil {
		err = waitErr
	}
	if err != nil {
		if msg := strings.TrimSpace(r.stderr.String()); msg != "" {
			err = fmt.Errorf("%w: %s", err, msg)
		}
	}
	r.err = err
}

func (r *recorder) push(j []byte) {
	select {
	case r.frames <- j:
		r.pushed++
	default:
		r.drop++
	}
}

// close finishes the file. No accepted frames removes whatever ffmpeg
// started. A failure after frames were written is returned to the caller.
func (r *recorder) close() error {
	close(r.frames)
	<-r.done
	if r.wrote == 0 {
		os.Remove(r.path)
		if r.pushed == 0 {
			fmt.Fprintf(os.Stderr, "omvcon: no frames recorded\n")
			return nil
		}
		fmt.Fprintf(os.Stderr, "omvcon: record: %v\n", r.err)
		return r.err
	}
	if r.err != nil {
		fmt.Fprintf(os.Stderr, "omvcon: record: %v\n", r.err)
		return r.err
	}
	if r.drop > 0 {
		fmt.Fprintf(os.Stderr, "omvcon: recorded %s (%d frames, %d dropped)\n", r.path, r.wrote, r.drop)
		return nil
	}
	fmt.Fprintf(os.Stderr, "omvcon: recorded %s (%d frames)\n", r.path, r.wrote)
	return nil
}

func viewCmd(path string, intr <-chan os.Signal, args []string) {
	fs := flag.NewFlagSet("view", flag.ExitOnError)
	addr := fs.String("http", "localhost:8080", "where to serve the page (\":8080\" for every interface)")
	open := fs.Bool("open", false, "open the page in the browser")
	debug := fs.Bool("debug", false, "print the script's cue and lock lines")
	record := fs.String("record", "", "write each frame to this mp4")
	fs.Parse(args)
	if fs.NArg() != 1 {
		flag.Usage()
		os.Exit(exitUsage)
	}
	code, err := os.ReadFile(fs.Arg(0))
	if err != nil {
		fail(exitUsage, err)
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		fail(exitUsage, err)
	}

	var rec *recorder
	if *record != "" {
		rec, err = startRecorder(*record)
		if err != nil {
			fail(exitPython, err)
		}
	}
	// exit calls os.Exit, so the file has to be closed on each path out.
	stopRecord := func() error {
		if rec == nil {
			return nil
		}
		err := rec.close()
		rec = nil
		return err
	}

	b := attach(path, intr)
	exc, err := b.exec(fmt.Sprintf(injectCode, strconv.Quote(omvviewSource)), os.Stdout, 10*time.Second)
	if err != nil {
		stopRecord()
		fail(exitUnreachable, err)
	}
	if exc != "" {
		fmt.Fprint(os.Stderr, exc)
		stopRecord()
		fail(exitPython, errors.New("could not install omvview on the board"))
	}

	h := &hub{next: make(chan struct{}), events: make(chan string, 32), track: "NA", debug: *debug, rec: rec}
	go http.Serve(ln, h.mux())
	url := "http://" + *addr + "/"
	if strings.HasPrefix(*addr, ":") {
		url = "http://localhost" + *addr + "/"
	}
	fmt.Fprintf(os.Stderr, "omvcon: view at %s\n", url)
	if *open {
		opener := "xdg-open"
		if runtime.GOOS == "darwin" {
			opener = "open"
		}
		exec.Command(opener, url).Start()
	}

	if err := b.start(withDebug(string(code), *debug)); err != nil {
		stopRecord()
		fail(exitUnreachable, err)
	}
	exc, err = h.demux(b, *fTimeout)
	h.summary()
	recErr := stopRecord()
	if err != nil {
		fail(exitUnreachable, err)
	}
	b.leaveRaw()
	if exc != "" {
		fmt.Fprint(os.Stderr, exc)
		exit(exitPython)
	}
	if recErr != nil {
		exit(exitPython)
	}
}

// demux reads the running script's output until the raw REPL's end of
// output, then hands the exception and prompt to finish. On Ctrl-C or the
// timeout it sends Ctrl-C, and again every second: omvview switches the
// interrupt off while it writes a packet, and a script that does not call
// omvview.poll() never sees the one that landed in stdin then.
func (h *hub) demux(b *board, timeout time.Duration) (exc string, err error) {
	start := time.Now()
	var stopAt, lastIntr time.Time
	for {
		for len(b.in) > 0 {
			i := 0
			for i < len(b.in) && b.in[i] != 0xff && b.in[i] != 0x04 {
				i++
			}
			os.Stdout.Write(h.noteText(b.in[:i]))
			b.in = b.in[i:]
			if len(b.in) == 0 {
				break
			}
			if b.in[0] == 0x04 {
				return b.finish(os.Stdout, 10*time.Second)
			}
			k := min(len(b.in), len(frameMagic))
			if !bytes.Equal(b.in[:k], frameMagic[:k]) {
				os.Stdout.Write(b.in[:1])
				h.noteText(b.in[:1])
				b.in = b.in[1:]
				continue
			}
			if len(b.in) < len(frameMagic)+8 {
				break
			}
			n := int(binary.LittleEndian.Uint32(b.in[5:]))
			sum := binary.LittleEndian.Uint32(b.in[9:])
			if n > maxFrame {
				fmt.Fprintf(os.Stderr, "omvcon: frame header with length %d, skipped\n", n)
				b.in = b.in[1:]
				continue
			}
			if len(b.in) < 13+n {
				break
			}
			j := bytes.Clone(b.in[13 : 13+n])
			b.in = b.in[13+n:]
			h.publish(j, crc32.ChecksumIEEE(j) == sum)
		}

		if stopAt.IsZero() {
			select {
			case <-b.intr:
				stopAt = time.Now()
			default:
				if timeout > 0 && time.Since(start) > timeout {
					fmt.Fprintf(os.Stderr, "omvcon: %v passed, stopping the board's code\n", timeout)
					stopAt = time.Now()
				}
			}
		}
		if !stopAt.IsZero() {
			if time.Since(stopAt) > 10*time.Second {
				return "", errors.New("the board did not stop within 10 s of a Ctrl-C")
			}
			if time.Since(lastIntr) > time.Second {
				b.p.send("\x03")
				lastIntr = time.Now()
			}
		}

		// Events stop at the Ctrl-C: what the script does not read before
		// it ends would be read as code by the raw REPL.
		for drained := false; !drained && stopAt.IsZero(); {
			select {
			case ev := <-h.events:
				if err := b.p.write([]byte(ev+"\n"), 500*time.Millisecond); err != nil {
					fmt.Fprintf(os.Stderr, "omvcon: event %q not sent, the script is not reading them: %v\n", ev, err)
				}
			default:
				drained = true
			}
		}

		if _, err := b.fill(); err != nil {
			return "", err
		}
	}
}

// hub holds the newest frame for the HTTP side, and the counters.
type hub struct {
	mu     sync.Mutex
	jpeg   []byte
	next   chan struct{} // closed when a new frame arrives
	w, h   int
	frames int
	bad    int
	bytes  int
	first  time.Time
	recent []time.Time // arrivals in the last 2 s, for the fps
	event  string
	events chan string // to the board, one line each
	text   []byte      // incomplete console line
	dx, dy int
	conf   int
	hasOff bool
	placed bool   // a drag box is waiting for Lock
	held   bool   // an object is saved
	track  string // NA, LOCKED, COASTING, or LOST
	debug  bool   // echo cue lines and the lock; the diff line is always echoed
	// Camera load from the `meter` line. Temperatures are tenths of a degree Celsius.
	cpuPct, memPct    int
	hasCPU, hasMem    bool
	tempHigh, tempLow int
	tempMean          int
	hasTemp           bool
	rec               *recorder
}

// noteText keeps the latest lock for the page and returns the lines to echo.
// The page lines (`state ...`, `temp ...`, `meter ...`) are never echoed. Cue
// lines and `dx=` / `track=` are echoed only while debug is set. The diff line
// is the MAVLink readout and is always echoed.
func (h *hub) noteText(p []byte) []byte {
	if len(p) == 0 {
		return nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.text = append(h.text, p...)
	var out []byte
	for {
		i := bytes.IndexByte(h.text, '\n')
		if i < 0 {
			break
		}
		line := h.text[:i]
		h.readOffset(line)
		if showLine(line, h.debug) {
			out = append(out, line...)
			out = append(out, '\n')
		}
		h.text = h.text[i+1:]
	}
	if len(h.text) > 4096 {
		h.text = h.text[:0]
	}
	if cap(h.text) > 1024 && len(h.text)*4 < cap(h.text) {
		h.text = append([]byte(nil), h.text...)
	}
	return out
}

func showLine(line []byte, debug bool) bool {
	if n := len(line); n > 0 && line[n-1] == '\r' {
		line = line[:n-1]
	}
	f := strings.Fields(string(line))
	if len(f) == 0 {
		return true
	}
	if f[0] == "state" || f[0] == "meter" || f[0] == "temp" {
		return false
	}
	if !debug && len(f) == 1 && strings.HasPrefix(f[0], "track=") {
		return false
	}
	if !debug && len(f) == 3 {
		_, ok1 := labeledInt(f[0], "dx")
		_, ok2 := labeledInt(f[1], "dy")
		_, ok3 := labeledInt(f[2], "confidence")
		if ok1 && ok2 && ok3 {
			return false
		}
	}
	if !debug && strings.Contains(string(line), "_us=") {
		return false
	}
	return true
}

func (h *hub) readOffset(line []byte) {
	if n := len(line); n > 0 && line[n-1] == '\r' {
		line = line[:n-1]
	}
	f := strings.Fields(string(line))
	if len(f) >= 3 && f[0] == "state" {
		placed, err1 := strconv.Atoi(f[1])
		held, err2 := strconv.Atoi(f[2])
		if err1 != nil || err2 != nil {
			return
		}
		h.placed = placed != 0
		h.held = held != 0
		if len(f) != 7 {
			return
		}
		dx, err1 := strconv.Atoi(f[3])
		dy, err2 := strconv.Atoi(f[4])
		conf, err3 := strconv.Atoi(f[5])
		if err1 != nil || err2 != nil || err3 != nil {
			return
		}
		h.dx, h.dy, h.conf = dx, dy, conf
		h.hasOff = true
		switch f[6] {
		case "NA", "LOCKED", "COASTING", "LOST":
			h.track = f[6]
		}
		return
	}
	if len(f) == 1 && strings.HasPrefix(f[0], "track=") {
		switch f[0][len("track="):] {
		case "NA", "LOCKED", "COASTING", "LOST":
			h.track = f[0][len("track="):]
		}
		return
	}
	if len(f) == 4 && f[0] == "temp" {
		h.readTemp(f)
		return
	}
	if len(f) == 6 && f[0] == "meter" {
		h.readMeter(f)
		return
	}
	if len(f) != 3 {
		return
	}
	dx, ok1 := labeledInt(f[0], "dx")
	dy, ok2 := labeledInt(f[1], "dy")
	conf, ok3 := labeledInt(f[2], "confidence")
	if !ok1 || !ok2 || !ok3 {
		return
	}
	h.dx, h.dy, h.conf = dx, dy, conf
	h.hasOff = true
}

// readTemp takes `temp high low mean`. Values are tenths of a degree Celsius.
// `na` clears the temperature fields. This line does not carry cpu or memory.
func (h *hub) readTemp(f []string) {
	high, highSet, ok1 := meterValue(f[1])
	low, lowSet, ok2 := meterValue(f[2])
	mean, meanSet, ok3 := meterValue(f[3])
	if !ok1 || !ok2 || !ok3 {
		return
	}
	h.hasTemp = highSet && lowSet && meanSet
	if h.hasTemp {
		h.tempHigh, h.tempLow, h.tempMean = high, low, mean
	}
}

// readMeter takes `meter cpu mem high low mean`. cpu and mem are percents.
// high, low, and mean are tenths of a degree Celsius. `na` clears that field.
// Printed by the tracker only while --debug is set.
func (h *hub) readMeter(f []string) {
	cpu, cpuSet, ok1 := meterValue(f[1])
	mem, memSet, ok2 := meterValue(f[2])
	high, highSet, ok3 := meterValue(f[3])
	low, lowSet, ok4 := meterValue(f[4])
	mean, meanSet, ok5 := meterValue(f[5])
	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 {
		return
	}
	h.hasCPU, h.cpuPct = cpuSet, cpu
	h.hasMem, h.memPct = memSet, mem
	h.hasTemp = highSet && lowSet && meanSet
	if h.hasTemp {
		h.tempHigh, h.tempLow, h.tempMean = high, low, mean
	}
}

func meterValue(token string) (value int, set, ok bool) {
	if token == "na" {
		return 0, false, true
	}
	n, err := strconv.Atoi(token)
	if err != nil {
		return 0, false, false
	}
	return n, true, true
}

func labeledInt(token, name string) (int, bool) {
	prefix := name + "="
	if !strings.HasPrefix(token, prefix) {
		return 0, false
	}
	n, err := strconv.Atoi(token[len(prefix):])
	if err != nil {
		return 0, false
	}
	return n, true
}

func (h *hub) publish(j []byte, ok bool) {
	var w, ht int
	if ok {
		cfg, err := jpeg.DecodeConfig(bytes.NewReader(j))
		ok = err == nil
		w, ht = cfg.Width, cfg.Height
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if !ok {
		h.bad++
		return
	}
	now := time.Now()
	if h.frames == 0 {
		h.first = now
	}
	h.frames++
	h.bytes += len(j)
	h.recent = append(h.recent, now)
	for len(h.recent) > 0 && now.Sub(h.recent[0]) > 2*time.Second {
		h.recent = h.recent[1:]
	}
	h.jpeg, h.w, h.h = j, w, ht
	if h.rec != nil {
		h.rec.push(j)
	}
	close(h.next)
	h.next = make(chan struct{})
}

// statusMap is the /status document. The caller holds h.mu.
// cpu and memory are percents. Temperatures are tenths of a degree Celsius.
func (h *hub) statusMap() map[string]any {
	fps := 0.0
	if n := len(h.recent); n > 1 && time.Since(h.recent[n-1]) < time.Second {
		fps = float64(n-1) / h.recent[n-1].Sub(h.recent[0]).Seconds()
	}
	var dx, dy, confidence any
	if h.hasOff {
		dx, dy, confidence = h.dx, h.dy, h.conf
	}
	track := h.track
	if track == "" {
		track = "NA"
	}
	var cpu, memory, tempHigh, tempLow, tempMean any
	if h.hasCPU {
		cpu = h.cpuPct
	}
	if h.hasMem {
		memory = h.memPct
	}
	if h.hasTemp {
		tempHigh, tempLow, tempMean = h.tempHigh, h.tempLow, h.tempMean
	}
	return map[string]any{
		"fps": fps, "frames": h.frames, "bad": h.bad, "w": h.w, "h": h.h, "event": h.event,
		"dx": dx, "dy": dy, "confidence": confidence,
		"placed": h.placed, "held": h.held, "track": track,
		"cpu": cpu, "memory": memory,
		"temp_high": tempHigh, "temp_low": tempLow, "temp_mean": tempMean,
	}
}

func (h *hub) summary() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.frames < 2 {
		fmt.Fprintf(os.Stderr, "omvcon: %d frames, %d bad\n", h.frames, h.bad)
		return
	}
	d := h.recent[len(h.recent)-1].Sub(h.first).Seconds()
	fmt.Fprintf(os.Stderr, "omvcon: %d frames (%dx%d) in %.1f s: %.1f fps, %.0f KB/s, %d bad\n",
		h.frames, h.w, h.h, d, float64(h.frames-1)/d, float64(h.bytes)/1024/d, h.bad)
}

func (h *hub) mux() *http.ServeMux {
	m := http.NewServeMux()
	m.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(viewPage))
	})
	m.HandleFunc("GET /frame.jpg", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		j := h.jpeg
		h.mu.Unlock()
		if j == nil {
			http.Error(w, "no frame yet", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("Cache-Control", "no-store")
		w.Write(j)
	})
	m.HandleFunc("GET /stream", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=omvframe")
		w.Header().Set("Cache-Control", "no-store")
		rc := http.NewResponseController(w)
		for {
			h.mu.Lock()
			j, next := h.jpeg, h.next
			h.mu.Unlock()
			if j != nil {
				fmt.Fprintf(w, "--omvframe\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", len(j))
				w.Write(j)
				w.Write([]byte("\r\n"))
				if rc.Flush() != nil {
					return
				}
			}
			select {
			case <-next:
			case <-r.Context().Done():
				return
			}
		}
	})
	m.HandleFunc("GET /status", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		s := h.statusMap()
		h.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(s)
	})
	m.HandleFunc("POST /event", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(io.LimitReader(r.Body, 128))
		f := strings.Fields(string(body))
		v := make([]int, 0, 4)
		for _, s := range f[min(1, len(f)):] {
			x, err := strconv.Atoi(s)
			if err != nil || x < 0 || x > 65535 {
				break
			}
			v = append(v, x)
		}
		var ev string
		switch {
		case len(f) == 1 && (f[0] == "select" || f[0] == "lock" || f[0] == "unlock"):
			ev = f[0]
		case len(f) == 3 && f[0] == "click" && len(v) == 2:
			ev = fmt.Sprintf("click %d %d", v[0], v[1])
		case len(f) == 5 && f[0] == "box" && len(v) == 4:
			ev = fmt.Sprintf("box %d %d %d %d", v[0], v[1], v[2], v[3])
		default:
			http.Error(w, `want "select", "lock", "unlock", "click X Y", or "box X Y W H"`, http.StatusBadRequest)
			return
		}
		select {
		case h.events <- ev:
		default:
			http.Error(w, "event queue full", http.StatusServiceUnavailable)
			return
		}
		h.mu.Lock()
		h.event = ev
		h.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	})
	return m
}

// The overlay maps the pointer from displayed to JPEG pixels; a drag of
// under 5 screen pixels counts as a click.
const viewPage = `<!doctype html>
<html><head><meta charset="utf-8"><title>omvcon view</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg: #f4f4f2; --fg: #222; --dim: #666; --mark: #00a0ff; }
@media (prefers-color-scheme: dark) { :root { --bg: #151515; --fg: #ddd; --dim: #888; } }
body { margin: 0; padding: 12px 16px; background: var(--bg); color: var(--fg);
  font: 13px ui-monospace, Menlo, monospace; }
#stage { display: flex; align-items: flex-start; gap: 24px; }
#wrap { position: relative; flex: 1 1 auto; min-width: 0; max-width: 1024px; }
#v { display: block; width: 100%; height: auto; background: #000; image-rendering: pixelated; }
#c { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; touch-action: none; }
#side { width: 280px; flex: none; }
.readout { display: grid; grid-template-columns: 1fr auto; column-gap: 16px; row-gap: 12px;
  margin: 0 0 18px; align-items: baseline; }
.readout .label { color: var(--dim); }
.readout .val { font-size: 22px; }
.readout .section { margin-top: 10px; }
#s { margin-top: 14px; white-space: pre-wrap; color: var(--dim); }
#hint { margin-top: 12px; color: var(--dim); }
button { margin: 0 8px 0 0; padding: 6px 12px; font: inherit; color: var(--fg);
  background: transparent; border: 1px solid var(--dim); }
@media (max-width: 720px) {
  #stage { flex-direction: column; }
  #side { width: 100%; }
}
</style></head><body>
<div id="stage">
<div id="wrap"><img id="v" src="/stream" alt=""><canvas id="c"></canvas></div>
<aside id="side">
<div class="readout">
  <span class="label">State</span><span class="val" id="st">NA</span>
  <span class="label">X axis</span><span class="val" id="dx">—</span>
  <span class="label">Y axis</span><span class="val" id="dy">—</span>
  <span class="label">Confidence level</span><span class="val" id="cf">—</span>
  <span class="label section">CPU</span><span class="val section" id="cpu">—</span>
  <span class="label section">Memory</span><span class="val section" id="mem">—</span>
  <span class="label section">Highest temp</span><span class="val section" id="thi">—</span>
  <span class="label">Lowest temp</span><span class="val" id="tlo">—</span>
  <span class="label">Mean temp</span><span class="val" id="tmu">—</span>
</div>
<div><button id="sel" type="button">Select</button><button id="lok" type="button">Lock</button><button id="ulk" type="button">Unlock</button></div>
<div id="hint">Lock uses the center box. Select, then drag, to choose another. Unlock drops it.</div>
<div id="s">waiting for the first frame</div>
</aside>
</div>
<script>
const v = document.getElementById('v'), c = document.getElementById('c'),
      s = document.getElementById('s'), g = c.getContext('2d'),
      dx = document.getElementById('dx'), dy = document.getElementById('dy'),
      cf = document.getElementById('cf'), st = document.getElementById('st'),
      cpu = document.getElementById('cpu'), mem = document.getElementById('mem'),
      thi = document.getElementById('thi'), tlo = document.getElementById('tlo'),
      tmu = document.getElementById('tmu');
let size = [0, 0], down = null, mark = null, armed = false, sawPlace = false;
const AIM = 48;

function dims() { return [v.naturalWidth || size[0], v.naturalHeight || size[1]]; }
function aimBox(w, h) {
  if (!w || !h) return null;
  const sw = Math.min(AIM, w), sh = Math.min(AIM, h);
  return [Math.floor((w - sw) / 2), Math.floor((h - sh) / 2), sw, sh];
}
function corners(x, y, w, h, k, q) {
  const px = x * k, py = y * q, pw = w * k, ph = h * q, arm = Math.min(pw, ph) / 4;
  g.beginPath();
  g.moveTo(px, py + arm); g.lineTo(px, py); g.lineTo(px + arm, py);
  g.moveTo(px + pw - arm, py); g.lineTo(px + pw, py); g.lineTo(px + pw, py + arm);
  g.moveTo(px + pw, py + ph - arm); g.lineTo(px + pw, py + ph); g.lineTo(px + pw - arm, py + ph);
  g.moveTo(px + arm, py + ph); g.lineTo(px, py + ph); g.lineTo(px, py + ph - arm);
  g.stroke();
}
function toImg(e) {
  const r = c.getBoundingClientRect(), [w, h] = dims();
  const x = Math.floor((e.clientX - r.left) * w / r.width), y = Math.floor((e.clientY - r.top) * h / r.height);
  return [Math.max(0, Math.min(w - 1, x)), Math.max(0, Math.min(h - 1, y)), e.clientX, e.clientY];
}
function draw() {
  const r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1, [w, h] = dims();
  c.width = r.width * d; c.height = r.height * d;
  g.clearRect(0, 0, c.width, c.height);
  if (c.width > 0 && c.height > 0) {
    const x = c.width / 2, y = c.height / 2, arm = 8 * d;
    g.lineWidth = 2 * d;
    g.strokeStyle = '#fff';
    g.beginPath();
    g.moveTo(x - arm, y); g.lineTo(x + arm, y);
    g.moveTo(x, y - arm); g.lineTo(x, y + arm);
    g.stroke();
    if (w && !(mark && mark.length === 4)) {
      const aim = aimBox(w, h);
      if (aim) corners(aim[0], aim[1], aim[2], aim[3], c.width / w, c.height / h);
    }
  }
  if (!mark || !w) return;
  const k = c.width / w, q = c.height / h;
  g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--mark');
  g.lineWidth = 2 * d;
  if (mark.length === 2) {
    const x = (mark[0] + 0.5) * k, y = (mark[1] + 0.5) * q;
    g.beginPath(); g.arc(x, y, 8 * d, 0, 2 * Math.PI); g.stroke();
  } else {
    g.strokeRect(mark[0] * k, mark[1] * q, mark[2] * k, mark[3] * q);
  }
}
function box(a, b) {
  const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]);
  return [x, y, Math.abs(a[0] - b[0]) + 1, Math.abs(a[1] - b[1]) + 1];
}
function post(line) {
  fetch('/event', {method: 'POST', body: line}).catch(() => {});
}
document.getElementById('sel').onclick = () => {
  armed = true; mark = null; draw(); post('select');
};
document.getElementById('lok').onclick = () => post('lock');
document.getElementById('ulk').onclick = () => {
  armed = false; sawPlace = false; mark = null; draw(); post('unlock');
};
c.addEventListener('pointerdown', e => {
  if (!armed || !dims()[0]) return;
  down = toImg(e); c.setPointerCapture(e.pointerId);
});
c.addEventListener('pointermove', e => { if (!down) return; mark = box(down, toImg(e)); draw(); });
c.addEventListener('pointerup', e => {
  if (!down) return;
  const p = toImg(e);
  if (Math.hypot(p[2] - down[2], p[3] - down[3]) < 5) {
    mark = [down[0], down[1]]; post('click ' + mark.join(' '));
  } else {
    mark = box(down, p); post('box ' + mark.join(' '));
  }
  down = null; draw();
});
window.addEventListener('resize', draw);
function reading(v) { return v === null || v === undefined ? '—' : String(v); }
function percent(v) { return v === null || v === undefined ? '—' : String(v) + '%'; }
function celsius(v) { return v === null || v === undefined ? '—' : (v / 10).toFixed(1) + '°C'; }
setInterval(() => {
  fetch('/status').then(r => r.json()).then(j => {
    size = [j.w, j.h];
    st.textContent = j.track || 'NA';
    dx.textContent = reading(j.dx);
    dy.textContent = reading(j.dy);
    cf.textContent = reading(j.confidence);
    cpu.textContent = percent(j.cpu);
    mem.textContent = percent(j.memory);
    thi.textContent = celsius(j.temp_high);
    tlo.textContent = celsius(j.temp_low);
    tmu.textContent = celsius(j.temp_mean);
    s.textContent = j.frames ? j.fps.toFixed(1) + ' fps  ' + j.w + 'x' + j.h + '  frames ' + j.frames +
      '  bad ' + j.bad + '  last event: ' + (j.event || '-') : 'waiting for the first frame';
    if (j.placed) sawPlace = true;
    if (sawPlace && j.held && !j.placed) {
      sawPlace = false; armed = false; mark = null;
    }
    draw();
  }).catch(() => { s.textContent = 'omvcon has stopped'; });
}, 500);
</script></body></html>
`
