# Project Title

Wearable ASL-to-speech glasses that translates ASL into spoken audio

## Team

| Name   | GitHub                                                 | Email                 |
| ------ | ------------------------------------------------------ | --------------------- |
| Name 1 | [@pringlessingles](https://github.com/pringlessingles) | newton.tran@sjsu.edu  |
| Name 2 | [@under-dogeey](https://github.com/under-dogeey)       | robert.trinh@sjsu.edu |
| Name 3 | [@ajimenez8203](https://github.com/ajimenez8203)       | aaron.jimenez@sjsu.edu|
| Name 4 | [@mkhantkk](https://github.com/mkhantkk)               | minkhant.koko@sjsu.edu|

**Advisor:** [Kaikai Liu]

---

## Problem Statement

About 1 million people in the US uses ASL as their main sign of communication, yet majority of the hearing people do not understand it. Human interpeters are often not available, therefore showing how much gap there is in communcation especially in medical emergency situations. 

## Solution

Our project is an _Offline_ ASL translation embedded in a pair of glasses. There is a camera that streams the visual through ESP32 to a Jetson Nano which runs a trained model to recongize the hand signs and compose a sentence using a local LLM. The translated sentence then goes to the wearer through text-to-speech audio with no internet required.

### Key Features

- Fully offline
- Wearable factor
- Feature 3

---

## Demo

[Link to demo video or GIF]

**Live Demo:** [URL if deployed]

---

## Screenshots

| Feature     | Screenshot                                   |
| ----------- | -------------------------------------------- |
| [Feature 1] | ![Screenshot](docs/screenshots/feature1.png) |
| [Feature 2] | ![Screenshot](docs/screenshots/feature2.png) |

---

## Tech Stack

| Category   | Technology                      | Justification
| ---------- | ----------                      | -------------
| Frontend   |  N/A                            | N/A
| Backend    |  Python server on Jetson Nano   | Capable of running AI recognition model
| Database   |  N/A                            | N/A
| Hardware   |  1. Waveshare ESP32-P4-WIFI6    | 1. H.264 encoding and wifi capabilities
|            |  2. Jetson Nano                 | 2. Price-to-performance is good         
| Deployment |  ESP-IDF, JetPack SDK           | Provides native CSI and H.264 driver support


---

## Getting Started

### Prerequisites
 - ESP-IDF v5.3.2
 - Python 3.8+
 - Jetson Nano (with JetPack SDK installed)
 - ESP32-P4-WIFI6
 - OV5647 camera module

### Installation

```bash
# Clone the repository
git clone https://github.com/SJSU-CMPE-195/group-project-team-narm.git
cd group-project-team-narm

# Set up ESP-IDF v5.3.2
 - Visit Espressif's website: https://docs.espressif.com/projects/idf-im-ui/en/latest/
 - Follow Espressif's install guide for your system

#Configure the ESP32-P4 target
```bash
idf.py set-target esp32p4

#Build and flash
```bash
idf.py build
idf.py -p <PORT> flash monitor

#Set up Jetson Nano
(Newton)

# Set up environment variables
cp .env.example .env
# Edit .env with your values

# Run database migrations (if applicable)
[migration command]
```

### Running the POC
1. Power on the esp32 with the ov5647 camera connected via MIPI-CSI.
2. Flash the firmware using 'idf.py flash' .
3. The esp32 captures video, encodes it as h.264, and streams it over wifi.
4. On the Jetson Nano, run the inference server to receive the stream and perform ASL translation.

### What's Next
- Optimize latency for real-time performance
- Design and prototype the physical glasses form factor
- Expand supported ASL vocabulary


### Running Locally

```bash
# Development mode
[dev command]

# The app will be available at http://localhost:XXXX
```

### Running Tests

```bash
[test command]
```

---

## API Reference

<details>
<summary>Click to expand API endpoints</summary>

| Method | Endpoint            | Description         |
| ------ | ------------------- | ------------------- |
| GET    | `/api/resource`     | Get all resources   |
| GET    | `/api/resource/:id` | Get resource by ID  |
| POST   | `/api/resource`     | Create new resource |
| PUT    | `/api/resource/:id` | Update resource     |
| DELETE | `/api/resource/:id` | Delete resource     |

</details>

---

## Project Structure

```
.
├── [folder]/           # Description
├── src/                # Source code files
├── tests/              # Test files
├── docs/               # Documentation files
└── README.md
```

---

## Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

- `Add user authentication endpoint`
- `Fix database connection timeout issue`
- `Update README with setup instructions`

---

## Acknowledgments

- [Resource/Library/Person]
- [Resource/Library/Person]

---

## License

This project is licensed under the <FILL IN> License - see the [LICENSE](LICENSE) file for details.

---

_CMPE 195A/B - Senior Design Project | San Jose State University | Spring 2026_
