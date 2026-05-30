package main

import (
	"fmt"
	"net"
	"testing"
	"time"
)

// TestMasterWorkerHandshake verifies the core NK -> GO protocol
func TestMasterWorkerHandshake(t *testing.T) {
	// 1. Setup a Test Hub (Master) on a random port
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to start test hub: %v", err)
	}
	defer ln.Close()
	testPort := ln.Addr().(*net.TCPAddr).Port
	addr := fmt.Sprintf("127.0.0.1:%d", testPort)

	// 2. Launch a mock worker in a goroutine
	done := make(chan bool)
	go func() {
		conn, err := net.Dial("tcp", addr)
		if err != nil {
			t.Errorf("Worker failed to connect: %v", err)
			return
		}
		defer conn.Close()

		buf := make([]byte, 2)

		// Expect NK
		_, err = conn.Read(buf)
		if err != nil || string(buf) != "NK" {
			t.Errorf("Worker did not receive NK correctly: %v", err)
		}

		// Expect GO
		_, err = conn.Read(buf)
		if err != nil || string(buf) != "GO" {
			t.Errorf("Worker did not receive GO correctly: %v", err)
		}

		done <- true
	}()

	// 3. Master Hub accepts connection
	conn, err := ln.Accept()
	if err != nil {
		t.Fatalf("Master failed to accept: %v", err)
	}
	defer conn.Close()

	// 4. Send signals
	conn.Write([]byte("NK"))
	time.Sleep(100 * time.Millisecond)
	conn.Write([]byte("GO"))

	// 5. Wait for worker to finish
	select {
	case <-done:
		// Success
	case <-time.After(2 * time.Second):
		t.Error("Test timed out waiting for worker handshake")
	}
}

// TestHeartbeat verifies that the master sends HB signals
func TestHeartbeat(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln.Close()
	addr := ln.Addr().String()

	// Mock Master sending HB
	go func() {
		conn, _ := ln.Accept()
		defer conn.Close()
		for i := 0; i < 3; i++ {
			conn.Write([]byte("HB"))
			time.Sleep(100 * time.Millisecond)
		}
	}()

	// Worker receiving HB
	conn, _ := net.Dial("tcp", addr)
	defer conn.Close()

	buf := make([]byte, 2)
	for i := 0; i < 3; i++ {
		_, err := conn.Read(buf)
		if err != nil || string(buf) != "HB" {
			t.Errorf("Failed to receive heartbeat %d: %v", i, err)
		}
	}
}
