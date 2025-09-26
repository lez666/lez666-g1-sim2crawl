import time
import sys

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
NETWORK_CARD_NAME = 'enxc8a362b43bfd'

class StateChecker:
    def __init__(self):
        self.low_state = None
        self.counter = 0

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        self.counter += 1
        
        # Print state info every 50 messages (about 1 second at 50Hz)
        if self.counter % 50 == 0:
            self.print_state_info()

    def print_state_info(self):
        if self.low_state is None:
            print("No state data received yet...")
            return
            
        print(f"\n=== Low State Info (Message #{self.counter}) ===")
        
        # Basic info
        print(f"Tick: {self.low_state.tick}")
        print(f"Mode Machine: {self.low_state.mode_machine}")
        
        # IMU state
        imu = self.low_state.imu_state
        print(f"IMU - RPY: {imu.rpy}")
        print(f"IMU - Quaternion: {imu.quaternion}")
        print(f"IMU - Gyroscope: {imu.gyroscope}")
        print(f"IMU - Accelerometer: {imu.accelerometer}")
        
      
        print("=" * 50)

if __name__ == '__main__':
    print("G1 State Checker - Printing low_state information")
    print("Press Ctrl+C to exit")
    
    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)


    checker = StateChecker()
    
    # Create subscriber
    lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_subscriber.Init(checker.LowStateHandler, 10)
    
    print("Waiting for state data...")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)