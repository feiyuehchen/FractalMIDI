# TouchDesigner Integration Guide

The FractalMIDI system now broadcasts generated notes in real-time via WebSocket, allowing seamless integration with TouchDesigner for immersive visual performances.

## Connection Details

- **Protocol**: WebSocket (ws://)
- **URL**: `ws://<SERVER_IP>:8000/ws/touchdesigner`
- **Format**: JSON

## Data Format

The system sends JSON messages with the following structure:

```json
{
    "type": "notes",
    "data": [
        {
            "pitch": 60,          // MIDI pitch (0-127)
            "start": 0.0,         // Start time in 16th notes
            "duration": 4.0,      // Duration in 16th notes
            "velocity": 0.8       // Normalized velocity (0.0-1.0)
        },
        ...
    ]
}
```

## TouchDesigner Setup

1.  **WebSocket DAT**
    -   Create a **WebSocket DAT** operator.
    -   Set **Network Address** to `localhost` (or server IP).
    -   Set **Port** to `8000`.
    -   Set **URL** to `/ws/touchdesigner`.
    -   Toggle **Active** to On.

2.  **Parse JSON**
    -   Attach a **Text DAT** to the WebSocket DAT's output.
    -   Use Python to parse the incoming JSON:
    
    ```python
    import json

    def onReceiveText(dat, rowIndex, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'notes':
                notes = data['data']
                # Process notes here (e.g., write to Table DAT)
                op('notes_table').clear()
                op('notes_table').appendRow(['pitch', 'start', 'duration', 'velocity'])
                for note in notes:
                    op('notes_table').appendRow([
                        note['pitch'], 
                        note['start'], 
                        note['duration'], 
                        note['velocity']
                    ])
        except Exception as e:
            print(e)
    ```

3.  **Visuals**
    -   Use the parsed data to drive **Instancing** in Geometry COMPs.
    -   Map `pitch` to Y-position or Color.
    -   Map `velocity` to Scale or Brightness.
    -   Map `start` to X-position (Timeline).

## Example Scene Ideas

1.  **Particle Rain**: Notes falling from the sky, size based on velocity.
2.  **Circular visualizer**: Notes arranged in a circle (pitch = angle), pulsing outward.
3.  **3D Piano Roll**: Traditional piano roll extruded into 3D space with glowing neon materials.
