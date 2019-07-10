# Elevator-Traffic-Monitor
Elevator Traffic Monitor collects and analyzes elevator usage data, providing feedback to elevator riders. Elevator riders are informed of decisions about how to reach their destination faster, taking the stairs or elevators. 
The Elevator Traffic Monitor is divided into three sub-systems: data collection, data analysis, and information distribution.  Data Collection: Data collection sub-system consists of a camera connected to a WiFi enabled microcontroller. It takes a snapshot of the elevator bank every 20 seconds, it about the time an elevator spends stopped on one floor and sends it to a server set up on a Google Compute Engine instance. The images are timestamped and saved to a folder on the instance. 
Data Analysis: Data Analysis system reads the position of the elevator from the images using an SVM model trained to recognize a specific type of elevator. Keep the position and timestamp pair in database and calculate the velocity by time of day. Compare the velocity with the pre-recorded time used to take stairs. So that a decision about taking elevator or stairs could be made. 
Information distribution: The decision depends on what floors the user is in. Bluetooth low energy (BLE) beacons broadcast Eddystone-URL frames that contain URLs tied to the floor the beacon is located. When a user accesses the webpage tied to the beacon nearest them, they receive relevant travel suggestions from their starting point.
