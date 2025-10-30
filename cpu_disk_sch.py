# app.py
# Run this file with: streamlit run app.py
# FIX: Cleaned all hidden/invisible characters (like &nbsp;) that cause IndentationErrors.
# FIX: Renamed 'Avg. Seek Time' to 'Avg Seek Time' (no dot) for consistency.

# Import necessary libraries
import streamlit as st  # For creating the web app
import pandas as pd     # For creating data tables (DataFrames)
import copy             # To make deep copies of process lists
import heapq            # For priority queues (SJF, SRTF, Priority)
from collections import deque # For the ready queue in Round Robin

# -------------------------------------------------------------------
# 1. PROCESS CLASS (Data Structure)
# -------------------------------------------------------------------

# This class defines a "Process" and stores all its properties.
class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=0):
        self.pid = pid                     # Process ID (e.g., "P1")
        self.arrival_time = arrival_time   # When the process arrives
        self.burst_time = burst_time       # Total CPU time needed
        self.priority = priority           # Priority value (lower = higher priority)
        
        # --- Variables for calculation ---
        # remaining_burst_time is used by preemptive algorithms (SRTF, RR)
        self.remaining_burst_time = burst_time 
        self.completion_time = 0           # When the process finishes
        self.turnaround_time = 0           # completion_time - arrival_time
        self.waiting_time = 0              # turnaround_time - burst_time
        self.response_time = -1            # When the process *first* gets the CPU
    
    # A helper function to convert this object into a dictionary.
    # This is very useful for creating Pandas DataFrames later.
    def to_dict(self):
        return {
            "PID": self.pid,
            "Arrival": self.arrival_time,
            "Burst": self.burst_time,
            "Priority": self.priority,
            "Completion": self.completion_time,
            "Turnaround": self.turnaround_time,
            "Waiting": self.waiting_time,
            "Response": self.response_time
        }

# -------------------------------------------------------------------
# 2. CPU SCHEDULING ALGORITHMS
# -------------------------------------------------------------------
# Each function takes a list of Process objects and returns:
# 1. The *completed* list of processes (with metrics calculated)
# 2. The Gantt chart data (a list of tuples: (pid, duration))
#
# Note: `copy.deepcopy()` is used so that each algorithm starts
# with a fresh, unmodified list of processes.

def fcfs_cpu(processes):
    # Sort processes by arrival time (First-Come, First-Serve)
    processes_sorted = sorted(copy.deepcopy(processes), key=lambda p: p.arrival_time)
    current_time, gantt_chart = 0, []
    
    for p in processes_sorted:
        # If the CPU is idle before this process arrives, add "Idle" block
        if current_time < p.arrival_time:
            gantt_chart.append(("Idle", p.arrival_time - current_time))
            current_time = p.arrival_time # Move time forward
            
        # Set response time (it's the same as waiting time for FCFS)
        p.response_time = current_time - p.arrival_time
        
        # Calculate metrics
        p.completion_time = current_time + p.burst_time
        p.turnaround_time = p.completion_time - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        
        # Add to Gantt chart and update time
        gantt_chart.append((p.pid, p.burst_time))
        current_time = p.completion_time
        
    return processes_sorted, gantt_chart

def sjf_cpu(processes): # Non-Preemptive
    # Start with processes sorted by arrival time
    processes_copy = sorted(copy.deepcopy(processes), key=lambda p: p.arrival_time)
    n = len(processes_copy)
    completed, current_time = 0, 0
    gantt_chart = []
    
    # This is a min-heap (priority queue) that will store (burst_time, pid, process_object)
    # It keeps the process with the *shortest burst time* at the top.
    ready_queue_heap = []
    process_idx = 0 # To track which processes have arrived
    
    while completed < n: # Loop until all processes are done
        # Add all processes to the ready queue that have arrived by the current_time
        while process_idx < n and processes_copy[process_idx].arrival_time <= current_time:
            p = processes_copy[process_idx]
            heapq.heappush(ready_queue_heap, (p.burst_time, p.pid, p))
            process_idx += 1
            
        if not ready_queue_heap:
            # If the ready queue is empty, CPU is idle.
            # Move time forward to the next process's arrival.
            if process_idx < n:
                idle_time = processes_copy[process_idx].arrival_time - current_time
                gantt_chart.append(("Idle", idle_time))
                current_time = processes_copy[process_idx].arrival_time
            continue # Go back to the top to add the newly arrived process
            
        # Pop the process with the shortest burst time from the heap
        burst_time, pid, p = heapq.heappop(ready_queue_heap)
        
        # Calculate metrics (this is non-preemptive, so it runs to completion)
        if p.response_time == -1:
            p.response_time = current_time - p.arrival_time
        p.completion_time = current_time + p.burst_time
        p.turnaround_time = p.completion_time - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        
        gantt_chart.append((p.pid, p.burst_time))
        current_time = p.completion_time
        completed += 1
        
    return processes_copy, gantt_chart

def srtf_cpu(processes): # Preemptive
    processes_copy = sorted(copy.deepcopy(processes), key=lambda p: p.arrival_time)
    n = len(processes_copy)
    completed, current_time = 0, 0
    gantt_chart = []
    
    # Min-heap based on *remaining* burst time
    ready_queue_heap = []
    process_idx = 0
    
    while completed < n:
        # Add all newly arrived processes to the heap
        while process_idx < n and processes_copy[process_idx].arrival_time <= current_time:
            p = processes_copy[process_idx]
            heapq.heappush(ready_queue_heap, (p.remaining_burst_time, p.pid, p))
            process_idx += 1
            
        if not ready_queue_heap:
            # CPU is idle
            if process_idx < n:
                idle_time = processes_copy[process_idx].arrival_time - current_time
                if idle_time > 0:
                    gantt_chart.append(("Idle", idle_time))
                    current_time = processes_copy[process_idx].arrival_time
            else:
                break # All processes are done
            continue
            
        # Pop the process with the *shortest remaining time*
        remaining_time, pid, p = heapq.heappop(ready_queue_heap)
        
        # Set response time if it's the first time running
        if p.response_time == -1:
            p.response_time = current_time - p.arrival_time
            
        # Add a 1-tick block to the Gantt chart
        if gantt_chart and gantt_chart[-1][0] == p.pid:
            # If the last block was also this process, just extend it
            gantt_chart[-1] = (p.pid, gantt_chart[-1][1] + 1)
        else:
            gantt_chart.append((p.pid, 1)) # Add new block of duration 1
            
        # Process for 1 time unit
        p.remaining_burst_time -= 1
        current_time += 1
        
        if p.remaining_burst_time == 0:
            # Process is finished
            p.completion_time = current_time
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            completed += 1
        else:
            # Process is not finished, put it back in the heap
            # Its remaining_burst_time is now smaller
            heapq.heappush(ready_queue_heap, (p.remaining_burst_time, p.pid, p))
            
    return processes_copy, gantt_chart

def rr_cpu(processes, time_quantum): # Preemptive
    processes_copy = sorted(copy.deepcopy(processes), key=lambda p: p.arrival_time)
    n = len(processes_copy)
    completed, current_time = 0, 0
    gantt_chart = []
    
    # Ready queue for Round Robin is a simple FIFO (First-In, First-Out) queue
    ready_queue = deque()
    process_idx = 0
    
    # Add initial processes to the queue
    while process_idx < n and processes_copy[process_idx].arrival_time <= current_time:
        ready_queue.append(processes_copy[process_idx])
        process_idx += 1
        
    while completed < n:
        if not ready_queue:
            # CPU is idle
            if process_idx < n:
                idle_time = processes_copy[process_idx].arrival_time - current_time
                gantt_chart.append(("Idle", idle_time))
                current_time = processes_copy[process_idx].arrival_time
                # Add all processes that arrived during the idle time
                while process_idx < n and processes_copy[process_idx].arrival_time <= current_time:
                    ready_queue.append(processes_copy[process_idx])
                    process_idx += 1
            else:
                break # All processes done
            continue
            
        # Get the next process from the front of the queue
        p = ready_queue.popleft()
        
        if p.response_time == -1:
            p.response_time = current_time - p.arrival_time
        
        # Run for the time quantum OR the remaining time, whichever is smaller
        exec_time = min(p.remaining_burst_time, time_quantum)
        gantt_chart.append((p.pid, exec_time))
        current_time += exec_time
        p.remaining_burst_time -= exec_time
        
        # IMPORTANT: Add processes that arrived *while this one was running*
        while process_idx < n and processes_copy[process_idx].arrival_time <= current_time:
            ready_queue.append(processes_copy[process_idx])
            process_idx += 1
            
        if p.remaining_burst_time == 0:
            # Process is finished
            p.completion_time = current_time
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            completed += 1
        else:
            # Process is not finished, put it back at the *end* of the queue
            ready_queue.append(p)
            
    return processes_copy, gantt_chart

def priority_cpu(processes): # Non-Preemptive
    processes_copy = sorted(copy.deepcopy(processes), key=lambda p: p.arrival_time)
    n, completed, current_time = len(processes_copy), 0, 0
    gantt_chart = []
    
    # Min-heap based on priority (lower number = higher priority)
    ready_queue_heap = []
    process_idx = 0
    
    while completed < n:
        # Add all arrived processes to the priority queue
        while process_idx < n and processes_copy[process_idx].arrival_time <= current_time:
            p = processes_copy[process_idx]
            heapq.heappush(ready_queue_heap, (p.priority, p.pid, p))
            process_idx += 1
            
        if not ready_queue_heap:
            # CPU is idle
            if process_idx < n:
                idle_time = processes_copy[process_idx].arrival_time - current_time
                gantt_chart.append(("Idle", idle_time))
                current_time = processes_copy[process_idx].arrival_time
            continue
            
        # Pop the process with the *highest priority* (lowest number)
        priority, pid, p = heapq.heappop(ready_queue_heap)
        
        if p.response_time == -1:
            p.response_time = current_time - p.arrival_time
            
        # Run to completion
        p.completion_time = current_time + p.burst_time
        p.turnaround_time = p.completion_time - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        
        gantt_chart.append((p.pid, p.burst_time))
        current_time = p.completion_time
        completed += 1
        
    return processes_copy, gantt_chart

# -------------------------------------------------------------------
# 3. DISK SCHEDULING ALGORITHMS
# -------------------------------------------------------------------
# Each function takes a list of requests and a starting head position.
# They return:
# 1. The seek_sequence (a list of cylinders visited)
# 2. The total_movement (total cylinders traversed)

def fcfs_disk(requests, head):
    requests_copy, total_movement = copy.deepcopy(requests), 0
    seek_sequence = [head] # Start at the head
    
    for req in requests_copy: # Process in the order they were given
        total_movement += abs(req - head) # Add distance
        head = req # Move head
        seek_sequence.append(head) # Record the stop
        
    return seek_sequence, total_movement

def sstf_disk(requests, head):
    requests_copy, total_movement = copy.deepcopy(requests), 0
    seek_sequence = [head]
    
    while requests_copy: # Loop until all requests are served
        # Find the request *closest* to the current head position
        next_req = min(requests_copy, key=lambda r: abs(r - head))
        
        total_movement += abs(next_req - head)
        head = next_req
        seek_sequence.append(head)
        requests_copy.remove(next_req) # Remove the served request
        
    return seek_sequence, total_movement

def scan_disk(requests, head, direction, max_cylinder):
    requests_copy, total_movement = sorted(copy.deepcopy(requests)), 0
    seek_sequence = [head]
    
    if direction == "right":
        # Find all requests to the right, in ascending order
        right = [r for r in requests_copy if r >= head]
        for req in right:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Go all the way to the end (max_cylinder)
        if head != max_cylinder:
            total_movement += abs(max_cylinder - head); head = max_cylinder
            if not right or max_cylinder != right[-1]: seek_sequence.append(head)
            
        # Find all requests to the left, in descending order (reverse)
        left = sorted([r for r in requests_copy if r < head], reverse=True)
        for req in left:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
    else: # direction == "left"
        # Find all requests to the left, in descending order
        left = sorted([r for r in requests_copy if r <= head], reverse=True)
        for req in left:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Go all the way to the beginning (cylinder 0)
        if head != 0:
            total_movement += abs(0 - head); head = 0
            if not left or 0 != left[-1]: seek_sequence.append(head)
            
        # Find all requests to the right, in ascending order
        right = sorted([r for r in requests_copy if r > head])
        for req in right:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
    return seek_sequence, total_movement

def c_scan_disk(requests, head, direction, max_cylinder):
    requests_copy, total_movement = sorted(copy.deepcopy(requests)), 0
    seek_sequence = [head]
    
    if direction == "right":
        # Service all requests to the right
        right = [r for r in requests_copy if r >= head]
        for req in right:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Go to the end
        if head != max_cylinder:
            total_movement += abs(max_cylinder - head); head = max_cylinder
            if not right or max_cylinder != right[-1]: seek_sequence.append(head)
            
        # Jump to the beginning (cylinder 0)
        total_movement += max_cylinder; head = 0
        if 0 not in requests_copy: seek_sequence.append(head)
        
        # Service remaining requests (which are all to the "right" of 0)
        remaining = [r for r in requests_copy if r < seek_sequence[1]]
        for req in remaining:
             total_movement += abs(req - head); head = req; seek_sequence.append(head)
    else: # direction == "left"
        # Service all requests to the left
        left = sorted([r for r in requests_copy if r <= head], reverse=True)
        for req in left:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Go to the beginning
        if head != 0:
            total_movement += abs(0 - head); head = 0
            if not left or 0 != left[-1]: seek_sequence.append(head)
            
        # Jump to the end (max_cylinder)
        total_movement += max_cylinder; head = max_cylinder
        if max_cylinder not in requests_copy: seek_sequence.append(head)
        
        # Service remaining requests (which are all to the "left" of max_cylinder)
        remaining = sorted([r for r in requests_copy if r > seek_sequence[1]], reverse=True)
        for req in remaining:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
    return seek_sequence, total_movement

def look_disk(requests, head, direction):
    # Same as SCAN, but does *not* go all the way to the ends (0 and max_cylinder)
    requests_copy, total_movement = sorted(copy.deepcopy(requests)), 0
    seek_sequence = [head]
    
    if direction == "right":
        # Go to the rightmost *request*
        right = [r for r in requests_copy if r >= head]
        for req in right:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Reverse direction and service the left requests
        left = sorted([r for r in requests_copy if r < head], reverse=True)
        for req in left:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
    else: # direction == "left"
        # Go to the leftmost *request*
        left = sorted([r for r in requests_copy if r <= head], reverse=True)
        for req in left:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Reverse direction and service the right requests
        right = sorted([r for r in requests_copy if r > head])
        for req in right:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
    return seek_sequence, total_movement

def c_look_disk(requests, head, direction):
    # Same as C-SCAN, but "jumps" to the first/last *request* instead of 0/max_cylinder
    requests_copy, total_movement = sorted(copy.deepcopy(requests)), 0
    seek_sequence = [head]
    
    if direction == "right":
        # Service requests to the right
        right = [r for r in requests_copy if r >= head]
        for req in right:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Jump to the *smallest* request
        left = [r for r in requests_copy if r < seek_sequence[1]] # Find requests not served
        if left:
            head = min(left) # Jump to smallest
            total_movement += abs(seek_sequence[-1] - head)
            seek_sequence.append(head)
            
            # Service remaining requests (still moving right)
            remaining = sorted([r for r in left if r > head])
            for req in remaining:
                total_movement += abs(req - head); head = req; seek_sequence.append(head)
    else: # direction == "left"
        # Service requests to the left
        left = sorted([r for r in requests_copy if r <= head], reverse=True)
        for req in left:
            total_movement += abs(req - head); head = req; seek_sequence.append(head)
            
        # Jump to the *largest* request
        right = [r for r in requests_copy if r > seek_sequence[1]] # Find requests not served
        if right:
            head = max(right) # Jump to largest
            total_movement += abs(seek_sequence[-1] - head)
            seek_sequence.append(head)
            
            # Service remaining requests (still moving left)
            remaining = sorted([r for r in right if r < head], reverse=True)
            for req in remaining:
                total_movement += abs(req - head); head = req; seek_sequence.append(head)
                
    return seek_sequence, total_movement

# -------------------------------------------------------------------
# 4. HELPER FUNCTIONS (Parsing, Metrics, Gantt)
# -------------------------------------------------------------------

# Calculates average metrics from a list of *completed* processes
def _calculate_cpu_metrics(processes):
    n = len(processes)
    if n == 0: return 0, 0, 0
    avg_tat = sum(p.turnaround_time for p in processes) / n
    avg_wt = sum(p.waiting_time for p in processes) / n
    avg_rt = sum(p.response_time for p in processes) / n
    return avg_tat, avg_wt, avg_rt

# Generates a text-based Gantt chart and counts context switches
def _get_gantt_text(gantt_chart):
    if not gantt_chart:
        return "", 0

    # Step 1: Merge contiguous blocks (e.g., [("P1", 1), ("P1", 1)] -> [("P1", 2)])
    # This is crucial for SRTF and RR to look clean.
    merged_chart = []
    last_pid = gantt_chart[0][0]
    current_duration = 0
    for pid, duration in gantt_chart:
        if pid == last_pid:
            current_duration += duration
        else:
            merged_chart.append((last_pid, current_duration))
            last_pid = pid
            current_duration = duration
    merged_chart.append((last_pid, current_duration)) # Add the final block

    # Step 2: Build the text chart and count switches
    line = "|"
    time_line = "0"
    current_time = 0
    switches = 0
    last_pid = None # Tracks the last *process* that was running
    
    for item, duration in merged_chart:
        # Create the bar text (e.g., "---P1---")
        bar_content = f"{item}"
        bar_len = max(len(bar_content) + 2, len(str(duration)) + 2) 
        bar = f"{item}".center(bar_len, "-")
        
        line += bar + "|"
        current_time += duration
        time_str = str(current_time)
        
        # Add padding to align the time stamp under the "|"
        padding = len(line) - len(time_line) - (len(time_str))
        if padding < 0:
            padding = 0
        
        time_line += " " * padding + time_str
        
        # --- Context Switch Logic ---
        if item == "Idle":
            last_pid = "Idle" # Reset on Idle
            continue
            
        # If this is the first process, set last_pid and don't count a switch
        if last_pid is None or last_pid == "Idle":
            last_pid = item
            continue
            
        # If the new item is different from the last *process*, count a switch
        if item != last_pid:
            switches += 1
        
        last_pid = item # Update the last running process

    return line + "\n" + time_line, switches


# -------------------------------------------------------------------
# 5. MAIN WEB APP UI
# -------------------------------------------------------------------

# Set the page configuration (title, layout)
st.set_page_config(page_title="OS Simulator", layout="wide")
st.title("🖥️ OS Process & Disk Scheduling Simulator")

# Create two tabs for the two modules
cpu_tab, disk_tab = st.tabs(["CPU Scheduling", "Disk Scheduling"])

# --- CPU Tab ---
with cpu_tab:
    st.header("CPU Scheduling Simulator")
    
    # `st.form` groups inputs and a button.
    # This prevents the app from re-running on every widget change.
    with st.form(key="cpu_form"):
        col1, col2 = st.columns([2, 1]) # Create a 2:1 ratio layout
        
        with col1:
            # A large text area for process input
            process_data_string = st.text_area(
                "Process Data (Arrival Time, Burst Time, Priority)",
                "0 8 3\n1 4 1\n2 5 2\n3 2 1", # Default example
                height=150,
                help="Enter one process per line. Format: AT BT PRIORITY"
            )
        
        with col2:
            # A number input for the time quantum
            time_quantum = st.number_input(
                "Time Quantum (for Round Robin)",
                min_value=1,
                value=4
            )
        
        # The button that submits the form
        submit_button = st.form_submit_button(label="Run CPU Algorithms")

    # This code only runs *after* the button is pressed
    if submit_button:
        try:
            # 1. Parse Input
            processes = []
            lines = process_data_string.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.strip().split()
                at, bt = int(parts[0]), int(parts[1])
                # Priority is optional, defaults to 0
                priority = int(parts[2]) if len(parts) > 2 else 0
                processes.append(Process(f"P{i+1}", at, bt, priority))
            
            if not processes:
                st.warning("Please enter at least one process.")
            else:
                # 2. Run Algorithms
                all_results = []       # To store detailed data for expanders
                comparison_data = []   # To store summary data for the main table/charts
                
                # Helper function to run an algorithm and store its results
                def run_and_get_metrics(name, algo_func, *args):
                    processes_done, gantt_raw = algo_func(*args)
                    tat, wt, rt = _calculate_cpu_metrics(processes_done)
                    gantt_text, cs = _get_gantt_text(gantt_raw)
                    
                    all_results.append({
                        "name": name, 
                        "data": processes_done, 
                        "gantt_text": gantt_text, 
                        "metrics": (tat, wt, rt), 
                        "cs": cs
                    })
                    comparison_data.append({
                        "Algorithm": name, 
                        "Avg TAT": tat, 
                        "Avg WT": wt, 
                        "Avg RT": rt, 
                        "Context Switches": cs
                    })

                # Run all 5 algorithms
                run_and_get_metrics("FCFS", fcfs_cpu, processes)
                run_and_get_metrics("SJF (Non-Preemptive)", sjf_cpu, processes)
                run_and_get_metrics("SRTF (Preemptive)", srtf_cpu, processes)
                run_and_get_metrics("Priority (Non-Preemptive)", priority_cpu, processes)
                run_and_get_metrics(f"Round Robin (Q={time_quantum})", rr_cpu, processes, time_quantum)

                # 3. Display Results
                st.subheader("📊 Algorithm Comparison")
                # Create a DataFrame from the summary data
                comp_df = pd.DataFrame(comparison_data).set_index("Algorithm")
                
                # Display the DataFrame as a formatted table
                st.dataframe(comp_df.style.format({
                    "Avg TAT": "{:.2f}", 
                    "Avg WT": "{:.2f}", 
                    "Avg RT": "{:.2f}",
                    "Context Switches": "{:d}"
                }))
                
                # Display the bar chart for TAT and WT
                st.subheader("📈 Performance Chart (Avg. TAT vs Avg. WT)")
                chart_df = comp_df[['Avg TAT', 'Avg WT']]
                st.bar_chart(chart_df)

                # Display the detailed results in expandable sections
                st.subheader("📑 Detailed Results")
                for res in all_results:
                    with st.expander(f"**{res['name']}**"):
                        tat, wt, rt = res['metrics']
                        # `st.metric` shows a large, formatted number
                        st.metric(label="Average Turnaround Time", value=f"{tat:.2f}")
                        st.metric(label="Average Waiting Time", value=f"{wt:.2f}")
                        st.metric(label="Average Response Time", value=f"{rt:.2f}")
                        st.metric(label="Total Context Switches", value=res['cs'])
                        
                        st.write("Gantt Chart:")
                        # `st.code` displays text in a fixed-width code block
                        st.code(res['gantt_text'], language="text")

                        st.write("Process Table:")
                        # Create a DataFrame from the list of completed process objects
                        df = pd.DataFrame([p.to_dict() for p in res['data']]).set_index("PID")
                        st.dataframe(df)
        
        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your input format.")


# --- Disk Tab ---
with disk_tab:
    st.header("Disk Scheduling Simulator")
    
    with st.form(key="disk_form"):
        col1, col2, col3 = st.columns(3) # Three equal columns
        with col1:
            head = st.number_input("Initial Head Position", min_value=0, value=53)
        with col2:
            max_cylinder = st.number_input("Maximum Cylinder", min_value=1, value=199)
        with col3:
            direction = st.selectbox("Initial Direction", ("right", "left"))
        
        requests_str = st.text_input("Disk Requests (space-separated)", "98 183 37 122 14 124 65 67")
        
        submit_button = st.form_submit_button(label="Run Disk Algorithms")
    
    if submit_button:
        try:
            # 1. Parse Input
            requests = [int(r) for r in requests_str.strip().split()]
            if not requests:
                st.warning("Please enter at least one disk request.")
            else:
                # 2. Run Algorithms
                all_results = []
                comparison_data = []

                algos = {
                    "FCFS": fcfs_disk(requests, head),
                    "SSTF": sstf_disk(requests, head),
                    "SCAN": scan_disk(requests, head, direction, max_cylinder),
                    "C-SCAN": c_scan_disk(requests, head, direction, max_cylinder),
                    "LOOK": look_disk(requests, head, direction),
                    "C-LOOK": c_look_disk(requests, head, direction)
                }

                # Loop through the results of all 6 algorithms
                for name, (sequence, movement) in algos.items():
                    # Calculate average seek time
                    avg_seek = movement / (len(sequence) - 1) if len(sequence) > 1 else 0
                    
                    all_results.append({
                        "name": name,
                        "sequence": " -> ".join(map(str, sequence)),
                        "movement": movement,
                        "avg_seek": avg_seek
                    })
                    comparison_data.append({
                        "Algorithm": name,
                        "Total Movement": movement,
                        "Avg Seek Time": avg_seek # Renamed to remove "."
                    })
                
                # 3. Display Results
                st.subheader("📊 Algorithm Comparison")
                comp_df = pd.DataFrame(comparison_data).set_index("Algorithm")
                st.dataframe(comp_df.style.format({
                    "Total Movement": "{:d}", 
                    "Avg Seek Time": "{:.2f}"
                }))
                
                st.subheader("📈 Performance Charts")
                
                st.write("**Total Head Movement**")
                # Use `y=` to explicitly plot the correct column
                st.bar_chart(comp_df, y='Total Movement')
                
                st.write("**Average Seek Time**")
                # Use `y=` to explicitly plot the correct column
                st.bar_chart(comp_df, y='Avg Seek Time')

                st.subheader("📑 Detailed Results")
                for res in all_results:
                    with st.expander(f"**{res['name']}**"):
                        st.metric(label="Total Head Movement", value=f"{res['movement']} cylinders")
                        st.metric(label="Average Seek Time", value=f"{res['avg_seek']:.2f}")
                        st.write("Seek Sequence:")
                        st.code(res['sequence'], language="text")

        except Exception as e:
            # Show a friendly error if input is bad (e.g., "hello" instead of numbers)
            st.error(f"An error occurred: {e}. Please check your input format.")