{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "non-default argument follows default argument (<ipython-input-1-2ec8692bf0a4>, line 12)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-1-2ec8692bf0a4>\"\u001b[0;36m, line \u001b[0;32m12\u001b[0m\n\u001b[0;31m    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300,seed):\u001b[0m\n\u001b[0m                ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m non-default argument follows default argument\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import torch \n",
    "import torch.nn as nn \n",
    "import torch.nn.functional as F\n",
    "\n",
    "def hidden_init(layer):\n",
    "    fan_in = layer.weight.data.size()[0]\n",
    "    lim = 1. / np.sqrt(fan_in)\n",
    "    return (-lim, lim)\n",
    "\n",
    "class Actor(nn.Module):\n",
    "    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):\n",
    "        super(Actor, self).__init__()\n",
    "        self.seed = torch.manual_seed(seed)\n",
    "        self.fc1 = nn.Linear(state_size, fc1_units)\n",
    "        self.fc2 = nn.Linear(fc1_units, fc2_units)\n",
    "        self.fc3 = nn.Linear(fc3_units, action_size)\n",
    "        self.bn1 = nn.BatchNorm1d(fc1_units)\n",
    "        self.bn2 = nn.BatchNorm1d(fc2_units)\n",
    "        self.reset_parameters()\n",
    "\n",
    "    def forward(self, state):\n",
    "        \"\"\"Build an actor (policy) network that maps states -> actions.\"\"\"\n",
    "        x = F.relu(self.bn1(self.fc1(state)))\n",
    "        x = F.relu(self.fc2(x))\n",
    "        return F.tanh(self.fc3(x))\n",
    "    \n",
    "    def reset_parameters(self):\n",
    "        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))\n",
    "        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))\n",
    "        # the final layer weights and biases of both the acotr and critic were initialized from a uniform distribution [-3x10^-3, 3x10^-3]\n",
    "        self.fc3.weight.data.uniform_(-3e-3, 3e-3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
