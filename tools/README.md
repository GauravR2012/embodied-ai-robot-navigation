# Local check tools

Verify the whole pipeline on your own machine BEFORE paying for a GPU
instance, then spend the paid time only on what needs the real models.

| Tool | Needs ROS | Needs GPU | Purpose |
|---|---|---|---|
| `mock_gateway.py` | no | no | Fake Vast.ai server (rule-based LLM + colour VLM), same HTTP contract |
| `preflight_check.py` | yes | no | Checks Nav2, topics, TF, gateway, and the real perception path |
| `vlm_batch_check.py` | no | real VLM | Scores saved frames against the real VLM per confidence threshold |

The mock listens on port 8000, the same port as the SSH tunnel and the
port `ui/gradio_app.py` uses. Run the mock OR the tunnel, never both.

## Phase 1: free dry run on your machine

Terminal 1 - Gazebo, your usual launch command.

Terminal 2 - Nav2 + localization, your usual launch command.

Terminal 3 - mock gateway (repo root):

    python3 tools/mock_gateway.py

Terminal 4 - preflight (repo root, ROS sourced). First without options:

    python3 tools/preflight_check.py

Then the real perception path, with the red box in the camera view and its
map position taken from your world SDF:

    python3 tools/preflight_check.py --target "red box" --expect X Y

Optionally check that Nav2 spinning works (the robot turns 15 deg and back):

    python3 tools/preflight_check.py --spin-test

Terminal 5 - the full UI against the mock:

    python3 -m ui.gradio_app

Try: `Go to Executor Test`, `Go to the red box`, turn the robot away and
repeat, `Go to the blue box` for a colour with no box (must not move).
The mock saves annotated images of what it "saw" in `/tmp/mock_gateway_debug`.

Useful mock options:

    python3 tools/mock_gateway.py --force-confidence 0.3   # low-confidence path
    python3 tools/mock_gateway.py --latency 3              # slow-model feel

## Phase 2: save labeled frames (still free)

With the mock stopped or running, in Terminal 4:

    python3 tools/preflight_check.py --skip-gateway --collect frames/

Drive the robot (teleop in another terminal) or move the box, then press
`p` + Enter when the target IS visible, `n` + Enter when it is NOT.
Aim for 10+ of each, including other-coloured boxes and empty views.

## Phase 3: the short paid session

1. Stop the mock (Ctrl+C in Terminal 3).
2. Start the Vast instance, start the gateway on it, open the SSH tunnel.
3. Quick health + real perception check:

       python3 tools/preflight_check.py --skip-ros
       python3 tools/preflight_check.py --target "red box" --expect X Y

4. Score the real VLM on the saved frames (no robot needed):

       python3 tools/vlm_batch_check.py --images frames/ --target "red box"

   Read the table: pick a confidence threshold with no false positives and
   set `OBJECT_MIN_CONFIDENCE` in `ui/gradio_app.py` to it.
5. Run `python3 -m ui.gradio_app` once through the real models, record it.
6. Stop the instance.

## What the mock can and cannot prove

It proves: topics, QoS, TF frames, depth -> map accuracy, goal generation,
`/spin`, Nav2 reaching the goal, the not-found behaviour, UI wiring.

It cannot prove: that Qwen follows the prompts, how good its boxes are, or
how trustworthy its confidence is. `vlm_batch_check.py` answers that.
