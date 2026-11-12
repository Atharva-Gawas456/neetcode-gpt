import numpy as np
import math
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = z - np.max(z)
        e = math.e
        sum = 0

        for i in range(0, len(z)):
            z[i] = e**z[i]
            sum += z[i]

        for i in range(0, len(z)):
            z[i] = round(z[i]/sum, 4)

        return z
        pass
