import numpy as np
class obs_history_gym:
    def __init__(self,num_obs,hist_len):
        self.hist_len = hist_len
        self.num_obs = num_obs
        self.total_obs = [0.]*self.num_obs*self.hist_len
    
    def update(self,new_obs):
        self.total_obs[self.num_obs:] = self.total_obs[:-self.num_obs]
        self.total_obs[:self.num_obs] = new_obs

        return np.array(self.total_obs)

class obs_history_lab:
    #term
    def __init__(self,num_obs,hist_len):
        self.hist_len = hist_len
        self.omeg_buffer = [0.]*3*self.hist_len
        self.gravity_orientation_buffer = [0.]*3*self.hist_len
        self.cmd_buffer = [0.]*3*self.hist_len
        self.position_buffer = [0.]*12*self.hist_len
        self.velocity_buffer = [0.]*12*self.hist_len
        self.last_cation_buffer = [0.]*12*self.hist_len
    
    def update(self,new_obs):
        self.omeg_buffer.extend(new_obs[:3])
        self.omeg_buffer = self.omeg_buffer[3:]
        
        self.gravity_orientation_buffer.extend(new_obs[3:6])
        self.gravity_orientation_buffer = self.gravity_orientation_buffer[3:]
        self.cmd_buffer.extend(new_obs[6:9])
        self.cmd_buffer = self.cmd_buffer[3:]
        self.position_buffer.extend(new_obs[9:21])
        self.position_buffer = self.position_buffer[12:]
        self.velocity_buffer.extend(new_obs[21:33])
        self.velocity_buffer = self.velocity_buffer[12:]
        self.last_cation_buffer.extend(new_obs[33:45])
        self.last_cation_buffer = self.last_cation_buffer[12:]

        total_obs = self.omeg_buffer + self.gravity_orientation_buffer + self.cmd_buffer  \
                    + self.position_buffer + self.velocity_buffer + self.last_cation_buffer
        
        # total_obs = np.array(total_obs)

        
        return np.array(total_obs)
