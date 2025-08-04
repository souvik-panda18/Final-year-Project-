# 🧠 NNPID-Controlled Line Follower Pick-and-Place Robot 🤖

## 📌 Overview

This project implements a smart **Line Follower Robot** integrated with a **pick-and-place manipulator**, powered by a **Neural Network-based PID (NNPID) controller**. The system dynamically adjusts its PID gains (`Kp`, `Ki`, `Kd`) using a trained neural network to achieve optimal performance in varying conditions, such as line curvature, speed variations, and load imbalances.

The manipulator arm is equipped with **3 Degrees of Freedom (DOF)**:
- Vertical movement (up/down)
- Horizontal movement (forward/backward)
- 360° base rotation

The robot is designed to navigate predefined paths, pick up objects using servo-actuated grippers, and place them at target zones autonomously.

---

## 🎯 Key Features

- 🚗 **Line Following**: Uses IR sensors to track black lines on a white surface.
- 🎯 **Real-time PID Tuning**: Neural network predicts optimal PID parameters based on current and past error states.
- 🦾 **3-DOF Pick-and-Place Arm**: Controlled via PWM signals and optimized via the NNPID model.
- 🧠 **Neural Network-Based Control**: Feedforward neural network predicts control gains dynamically, improving response and stability.
- 🔧 **Modular Design**: Clean separation of robot chassis control and manipulator logic.
- 📈 **Adaptive Behavior**: Reacts to disturbances (e.g., path curvature or uneven weights) with improved stability.

---

## 🛠 Technologies Used

- **Arduino UNO / Mega**
- **Python** (for training and prototyping NNs)
- **C++ (Arduino IDE)** for real-time control
- **Servo Motors** for arm control
- **DC Motors + Motor Driver (L298N)**
- **IR Sensors** for line detection
- **Neural Network** (Custom feedforward model with 1–2 hidden layers)
- **PID Control Logic** (Implemented with dynamic gains)

---

## 📦 Applications

- Industrial pick-and-place automation
- Smart warehouse robotics
- Human-assistive mobile manipulators
- Educational robotics with AI/ML integration

---

> 🔍 *A full explanation of the neural network architecture, control flow, and PID computation logic is provided in the documentation folder.*

---

Let me know if you also want:
- Diagram of the control system
- Training notebook link
- Video demo link (if you have one)
- Project paper / report summary

I can expand the README into a full version with usage, installation, wiring diagrams, etc. too.
