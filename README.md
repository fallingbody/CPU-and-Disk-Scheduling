# 🖥️ OS Process & Disk Scheduling Simulator

An interactive web app built with Streamlit to simulate, compare, and visualize common CPU and Disk Scheduling algorithms for Operating Systems.

This tool allows students and enthusiasts to input custom processes or disk requests and instantly see how different algorithms perform, complete with metrics, Gantt charts, and comparison graphs.

## Features

### 1. CPU Scheduling Simulator
Simulates the following algorithms:
* First-Come, First-Serve (FCFS)
* Shortest Job First (SJF) (Non-Preemptive)
* Shortest Remaining Time First (SRTF) (Preemptive)
* Priority Scheduling (Non-Preemptive)
* Round Robin (RR)

**Provides:**
* 📊 **Comparison Table:** A summary of Avg. Turnaround Time, Avg. Waiting Time, Avg. Response Time, and Total Context Switches.
* 📈 **Performance Graph:** A bar chart comparing the Average Turnaround Time and Average Waiting Time for all algorithms.
* 📑 **Detailed Results:** An expandable view for each algorithm showing:
    * Key performance metrics.
    * A text-based Gantt chart.
    * A full process table with completion, turnaround, and waiting times.

### 2. Disk Scheduling Simulator
Simulates the following algorithms:
* First-Come, First-Serve (FCFS)
* Shortest Seek Time First (SSTF)
* SCAN (Elevator)
* C-SCAN (Circular SCAN)
* LOOK
* C-LOOK

**Provides:**
* 📊 **Comparison Table:** A summary of Total Head Movement and Average Seek Time.
* 📈 **Performance Graphs:** Bar charts visualizing the Total Head Movement and Average Seek Time for all algorithms.
* 📑 **Detailed Results:** An expandable view for each algorithm showing:
    * Total head movement and average seek time.
    * The complete seek sequence.


## Screenshots
<img width="1746" height="772" alt="CPU Scheduling-1" src="https://github.com/user-attachments/assets/348298bb-22a6-4c7a-812b-4444f815c4af" />
<img width="1761" height="815" alt="CPU Scheduling-2 " src="https://github.com/user-attachments/assets/cb5c2bee-06e3-4ccb-85d2-8921a7c5e891" />
<img width="1739" height="930" alt="CPU Scheduling-3" src="https://github.com/user-attachments/assets/1734ca12-2583-4cf7-bb52-a7bd73baba34" />
<img width="1737" height="822" alt="Disk Scheduling-1" src="https://github.com/user-attachments/assets/acb51d7b-96ae-4291-a868-6a53a4f7ca92" />
<img width="1730" height="882" alt="Disk Scheduling-2" src="https://github.com/user-attachments/assets/adfdc926-53a7-4d08-b956-d675716f1975" />
<img width="1744" height="922" alt="Disk Scheduling-3" src="https://github.com/user-attachments/assets/68fe14dc-b9fe-46d6-b76d-193de8fe73b7" />

## How to Run

1.  **Clone the repository (or download the `cpu_disk_sch.py` file).**

2.  **Install the required dependencies:**
    ```bash
    pip install streamlit pandas
    ```

3.  **Run the application:**
    Open your terminal in the project's folder and run:
    ```bash
    streamlit run app.py
    ```

4.  **View the app:**
    Streamlit will automatically open the simulator in your web browser.
