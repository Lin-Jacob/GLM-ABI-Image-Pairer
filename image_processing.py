import os
import re
import datetime

# POSTPONE FOR NOW

class pairs:
  def __init__(self):
    self.pair = tuple
    self.date = datetime
    self.time = datetime
  
  def get_pair(self):
    return self.pair
  
  def get_date(self):
    return self.date
  
  def get_time(self):
    return self.time

class process_image:
  def __init__(self, directory):
    self.directory = directory
    self.files = os.listdir(directory)
    self.pairs = list
    self.process_files()
  
  def process_files(self):
    abi_files = []
    glm_files = []
    
    for file in self.files:
      if "ABI" in file:
        abi_files.append(file)
      elif "GLM" in file:
        glm_files.append(file)
        
    abi_pattern = re.compile(r"G16_ABI_B03_s(\d+)_e(\d+)_x([-?\d]+)y([-?\d]+)\.nc")
    glm_pattern = re.compile(r"G16_GLM_(\d+)_(\d+)_(\d+)_(\d+)_x([-?\d]+)y([-?\d]+)\.nc")

    for abi in abi_files:
      abi_match = abi_pattern.match(abi)
      if abi_match:
        s_time, e_time, x_coord, y_coord = abi_match.groups()
        print(s_time, e_time, x_coord, y_coord)
      
    for glm in glm_files:
      glm_match = glm_pattern.match(glm)
      if glm_match:
        s_year, s_month, s_day, s_time, x_coord, y_coord = glm_match.groups()
        print(s_year, s_month, s_day, s_time, x_coord, y_coord)
    
  def get_pairs(self):
    return self.pairs
      
  
directory = "data"
processed_image = process_image(directory)
