import numpy as np
import pickle

file_path = '/home/ar0058/Ash/repo/model_predictive_control/src/trajectory/'
trajectories = []
for i in range(1, 20):
    if i < 10:
        j = '0'+str(i)
    with open('%sjoint_angles%s.pickle' %(file_path, j), 'rb') as LmdR:
        ja = pickle.loads(LmdR.read())
        ja = np.array(ja, dtype=float)
        trajectories.append(ja)

with open('%strajectories.pickle' %file_path, 'wb') as f1:
    f1.write(pickle.dumps(trajectories))
with open('%strajectories.pickle' %file_path, 'rb') as f2:
    joint_angles = pickle.loads(f2.read())

print('hi')