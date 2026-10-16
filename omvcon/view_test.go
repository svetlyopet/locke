// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026 lvd

package main

import (
	"bytes"
	"image"
	"image/jpeg"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadOffset(t *testing.T) {
	h := &hub{}
	h.readOffset([]byte("dx=-27 dy=-24 confidence=23"))
	if !h.hasOff || h.dx != -27 || h.dy != -24 || h.conf != 23 {
		t.Fatalf("offset = %d %d %d hasOff=%v", h.dx, h.dy, h.conf, h.hasOff)
	}

	h.readOffset([]byte("appearance=41 appearance_us=62 unique=na unique_us=31"))
	if h.dx != -27 || h.dy != -24 || h.conf != 23 {
		t.Fatalf("cue line changed the offset to %d %d %d", h.dx, h.dy, h.conf)
	}

	h.readOffset([]byte("state 1 0"))
	if !h.placed || h.held {
		t.Fatalf("placed=%v held=%v", h.placed, h.held)
	}
	if h.dx != -27 || h.dy != -24 || h.conf != 23 {
		t.Fatalf("state line changed the offset to %d %d %d", h.dx, h.dy, h.conf)
	}

	h.readOffset([]byte("track=COASTING"))
	if h.track != "COASTING" {
		t.Fatalf("track = %q", h.track)
	}
	if h.dx != -27 || h.dy != -24 || h.conf != 23 {
		t.Fatalf("track line changed the offset to %d %d %d", h.dx, h.dy, h.conf)
	}
	h.readOffset([]byte("track=MAYBE"))
	if h.track != "COASTING" {
		t.Fatalf("unknown track replaced COASTING with %q", h.track)
	}
	h.readOffset([]byte("track=HIDDEN"))
	if h.track != "COASTING" {
		t.Fatalf("old HIDDEN label replaced COASTING with %q", h.track)
	}

	h.readOffset([]byte("state 0 1 -27 -24 23 COASTING"))
	if !h.hasOff || h.dx != -27 || h.dy != -24 || h.conf != 23 || h.track != "COASTING" || h.placed || !h.held {
		t.Fatalf("state lock = %d %d %d %q placed=%v held=%v", h.dx, h.dy, h.conf, h.track, h.placed, h.held)
	}

	h.readOffset([]byte("diff x=12 y=-4"))
	if h.dx != -27 || h.dy != -24 || h.conf != 23 {
		t.Fatalf("diff line changed the offset to %d %d %d", h.dx, h.dy, h.conf)
	}

	h.readOffset([]byte("-27 -24 23"))
	if h.dx != -27 || h.dy != -24 || h.conf != 23 {
		t.Fatalf("bare numbers changed the offset to %d %d %d", h.dx, h.dy, h.conf)
	}
}

func TestReadMeter(t *testing.T) {
	h := &hub{}
	h.readOffset([]byte("dx=1 dy=2 confidence=3"))
	h.readOffset([]byte("meter 42 61 362 181 248"))
	if !h.hasCPU || h.cpuPct != 42 || !h.hasMem || h.memPct != 61 || !h.hasTemp {
		t.Fatalf("meter flags cpu=%v mem=%v temp=%v", h.hasCPU, h.hasMem, h.hasTemp)
	}
	if h.tempHigh != 362 || h.tempLow != 181 || h.tempMean != 248 {
		t.Fatalf("meter temps = %d %d %d", h.tempHigh, h.tempLow, h.tempMean)
	}
	if h.dx != 1 || h.dy != 2 || h.conf != 3 {
		t.Fatalf("meter changed the offset to %d %d %d", h.dx, h.dy, h.conf)
	}

	s := h.statusMap()
	if s["cpu"] != 42 || s["memory"] != 61 || s["temp_high"] != 362 || s["temp_low"] != 181 || s["temp_mean"] != 248 {
		t.Fatalf("status = %v", s)
	}

	h.readOffset([]byte("meter na 10 na na na"))
	if h.hasCPU || h.hasTemp || !h.hasMem || h.memPct != 10 {
		t.Fatalf("cleared meter = cpu %v mem %d/%v temp %v", h.hasCPU, h.memPct, h.hasMem, h.hasTemp)
	}
	s = h.statusMap()
	if s["cpu"] != nil || s["temp_high"] != nil || s["memory"] != 10 {
		t.Fatalf("cleared status = %v", s)
	}

	h.readOffset([]byte("meter -3 0 -50 10 0"))
	if h.cpuPct != -3 || h.memPct != 0 || h.tempHigh != -50 || h.tempLow != 10 || h.tempMean != 0 {
		t.Fatalf("signed meter = %d %d %d %d %d", h.cpuPct, h.memPct, h.tempHigh, h.tempLow, h.tempMean)
	}

	h.readOffset([]byte("meter bad 1 2 3 4"))
	if h.cpuPct != -3 || h.memPct != 0 {
		t.Fatalf("bad meter replaced the load with %d %d", h.cpuPct, h.memPct)
	}

	h = &hub{}
	h.readOffset([]byte("temp 362 181 248"))
	if h.hasCPU || h.hasMem || !h.hasTemp {
		t.Fatalf("temp line set load cpu=%v mem=%v temp=%v", h.hasCPU, h.hasMem, h.hasTemp)
	}
	if h.tempHigh != 362 || h.tempLow != 181 || h.tempMean != 248 {
		t.Fatalf("temp line = %d %d %d", h.tempHigh, h.tempLow, h.tempMean)
	}
	s = h.statusMap()
	if s["cpu"] != nil || s["memory"] != nil || s["temp_high"] != 362 || s["temp_mean"] != 248 {
		t.Fatalf("temp status = %v", s)
	}
	h.readOffset([]byte("temp na na na"))
	if h.hasTemp {
		t.Fatal("temp na left the temperatures set")
	}
}

func TestNoteTextEchoesTheDiffLine(t *testing.T) {
	h := &hub{}
	got := h.noteText([]byte("state 1 0 12 -4 93 LOCKED\ntemp 362 181 248\nappearance=80 appearance_us=4\ndx=12 dy=-4 confidence=93\ntrack=LOCKED\ndiff x=12 y=-4\n"))
	if string(got) != "diff x=12 y=-4\n" {
		t.Fatalf("shown = %q", got)
	}
	if h.hasCPU || !h.hasTemp || h.tempMean != 248 {
		t.Fatalf("page missed the temperatures: cpu %v mean %d/%v", h.hasCPU, h.tempMean, h.hasTemp)
	}
	if !h.placed || h.held || h.dx != 12 || h.dy != -4 || h.conf != 93 || h.track != "LOCKED" {
		t.Fatalf("page missed the hidden lines: placed=%v held=%v dx=%d dy=%d conf=%d track=%s", h.placed, h.held, h.dx, h.dy, h.conf, h.track)
	}

	h.debug = true
	got = h.noteText([]byte("meter 7 8 9 10 11\nappearance=80 appearance_us=4\ndx=1 dy=2 confidence=3\ntrack=NA\ndiff x=1 y=2\n"))
	if string(got) != "appearance=80 appearance_us=4\ndx=1 dy=2 confidence=3\ntrack=NA\ndiff x=1 y=2\n" {
		t.Fatalf("debug shown = %q", got)
	}
}

func TestWithDebug(t *testing.T) {
	code := "print(1)\n"
	if withDebug(code, false) != code {
		t.Fatal("debug off changed the script")
	}
	got := withDebug(code, true)
	if !strings.HasPrefix(got, "import sys\n") || !strings.HasSuffix(got, code) || !strings.Contains(got, `"--debug"`) {
		t.Fatalf("debug script = %q", got)
	}
}

func TestFFmpegArgs(t *testing.T) {
	got := strings.Join(ffmpegArgs("assets/track.mp4"), " ")
	want := strings.Join([]string{
		"-y",
		"-loglevel", "error",
		"-f", "image2pipe",
		"-c:v", "mjpeg",
		"-framerate", "12",
		"-i", "pipe:0",
		"-an",
		"-vf", "scale=1024:768:flags=neighbor",
		"-c:v", "libx264",
		"-pix_fmt", "yuv420p",
		"-movflags", "+faststart",
		"assets/track.mp4",
	}, " ")
	if got != want {
		t.Fatalf("ffmpeg args = %q", got)
	}
}

func TestRecorderDropsAFullQueue(t *testing.T) {
	r := &recorder{frames: make(chan []byte, 1)}
	r.push([]byte("a"))
	r.push([]byte("b"))
	if r.pushed != 1 || r.drop != 1 {
		t.Fatalf("pushed=%d drop=%d", r.pushed, r.drop)
	}
}

func TestRecorderWritesAnMP4(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg is not on PATH")
	}
	path := filepath.Join(t.TempDir(), "track.mp4")
	rec, err := startRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	rec.push(tinyJPEG(t))
	rec.push(tinyJPEG(t))
	if err := rec.close(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() == 0 {
		t.Fatal("empty mp4")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var hdr [12]byte
	if _, err := f.Read(hdr[:]); err != nil {
		t.Fatal(err)
	}
	if string(hdr[4:8]) != "ftyp" {
		t.Fatalf("header = %q", hdr)
	}
}

func TestRecorderDropsAnEmptySession(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg is not on PATH")
	}
	path := filepath.Join(t.TempDir(), "track.mp4")
	rec, err := startRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("empty recording left a file: %v", err)
	}
}

func tinyJPEG(t *testing.T) []byte {
	t.Helper()
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for i := range img.Pix {
		img.Pix[i] = uint8(i)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}
