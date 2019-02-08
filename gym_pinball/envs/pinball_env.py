import io as StringIO
import numpy as np
from itertools import tee

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from . import util

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Pierre-Luc Bacon",  # author of the original version
              "Austin Hays",# adapted for RLPy and TKinter
              "Peter Vrancx"] #adapted for OpenAI gym

BOX_CFG="""ball 0.02
target 0.9 0.2 0.04
start 0.2 0.9

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0

polygon 0.45 0.45 0.55 0.45 0.55 0.55 0.45 0.55"""

EMPTY_CFG="""ball 0.02
target 0.9 0.2 0.04
start 0.2 0.9

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0"""

HARD_CFG="""ball 0.015
target 0.5 0.06 0.04
start 0.055 0.95

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0
polygon 0.034 0.852 0.106 0.708 0.33199999999999996 0.674 0.17599999999999996 0.618 0.028 0.718
polygon 0.15 0.7559999999999999 0.142 0.93 0.232 0.894 0.238 0.99 0.498 0.722
polygon 0.8079999999999999 0.91 0.904 0.784 0.7799999999999999 0.572 0.942 0.562 0.952 0.82 0.874 0.934
polygon 0.768 0.814 0.692 0.548 0.594 0.47 0.606 0.804 0.648 0.626
polygon 0.22799999999999998 0.5760000000000001 0.39 0.322 0.3400000000000001 0.31400000000000006 0.184 0.456
polygon 0.09 0.228 0.242 0.076 0.106 0.03 0.022 0.178
polygon 0.11 0.278 0.24600000000000002 0.262 0.108 0.454 0.16 0.566 0.064 0.626 0.016 0.438
polygon 0.772 0.1 0.71 0.20599999999999996 0.77 0.322 0.894 0.09600000000000002 0.8039999999999999 0.17600000000000002
polygon 0.698 0.476 0.984 0.27199999999999996 0.908 0.512
polygon 0.45 0.39199999999999996 0.614 0.25799999999999995 0.7340000000000001 0.438
polygon 0.476 0.868 0.552 0.8119999999999999 0.62 0.902 0.626 0.972 0.49 0.958
polygon 0.61 0.014000000000000002 0.58 0.094 0.774 0.05000000000000001 0.63 0.054000000000000006
polygon 0.33399999999999996 0.014 0.27799999999999997 0.03799999999999998 0.368 0.254 0.7 0.20000000000000004 0.764 0.108 0.526 0.158
polygon 0.294 0.584 0.478 0.626 0.482 0.574 0.324 0.434 0.35 0.39 0.572 0.52 0.588 0.722 0.456 0.668
"""

MEDIUM_CFG="""ball 0.02
target 0.9 0.2 0.04
start 0.2 0.9

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0

polygon 0.09 0.228 0.242 0.076 0.106 0.03 0.022 0.178
polygon 0.33399999999999996 0.014 0.27799999999999997 0.03799999999999998 0.368 0.254 0.7 0.20000000000000004 0.764 0.108 0.526 0.158
polygon 0.034 0.852 0.106 0.708 0.33199999999999996 0.674 0.17599999999999996 0.618 0.028 0.718
polygon 0.45 0.39199999999999996 0.614 0.25799999999999995 0.7340000000000001 0.438
polygon 0.33399999999999996 0.014 0.27799999999999997 0.03799999999999998 0.368 0.254 0.7 0.20000000000000004 0.764 0.108 0.526 0.158
polygon 0.294 0.584 0.478 0.626 0.482 0.574 0.324 0.434 0.35 0.39 0.572 0.52 0.588 0.722 0.456 0.668 """

SIMPLE_CFG="""ball 0.02
target 0.9 0.2 0.04
start 0.2 0.9

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0

polygon 0.35 0.4 0.45 0.55 0.43 0.65 0.3 0.7 0.45 0.7 0.5 0.6 0.45 0.35
polygon 0.2 0.6 0.25 0.55 0.15 0.5 0.15 0.45 0.2 0.3 0.12 0.27 0.075 0.35 0.09 0.55
polygon 0.3 0.8 0.6 0.75 0.8 0.8 0.8 0.9 0.6 0.85 0.3 0.9
polygon 0.8 0.7 0.975 0.65 0.75 0.5 0.9 0.3 0.7 0.35 0.63 0.65
polygon 0.6 0.25 0.3 0.07 0.15 0.175 0.15 0.2 0.3 0.175 0.6 0.3
polygon 0.75 0.025 0.8 0.24 0.725 0.27 0.7 0.025"""

CFGS = [EMPTY_CFG, BOX_CFG, SIMPLE_CFG, MEDIUM_CFG, HARD_CFG]

class PinBallEnv(gym.core.Env):
    metadata = {'render.modes': ['human']}

    """
    The goal of this domain is to maneuver a small ball on a plate into a hole.
    The plate may contain obstacles which should be avoided.

    **STATE:**
        The state is given by a 4-dimensional vector, consisting of position and
        velocity of the ball.

    **ACTIONS:**
        There are 5 actions, standing for slanting the  plat in x or y direction
        or a horizontal position
        of the plate.

    **REWARD:**
        Slanting the plate costs -4 reward in addition to -1 reward for each timestep.
        When the ball reaches the hole, the agent receives 10000 units of reward.

    **REFERENCE:**

    .. seealso::
        G.D. Konidaris and A.G. Barto:
        *Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining.*
        Advances in Neural Information Processing Systems 22, pages 1015-1023, December 2009.
    """
    #: default location of config files shipped with rlpy

    def __init__(self, noise=.1, episodeCap=1000,
                 configuration=2, infinite=False):
        """
        configuration:
            location of the configuration file
        episodeCap:
            maximum length of an episode
        noise:
            with probability noise, a uniformly random action is executed
        infinite (bool):
            use non-terminating variation that resets ball to random loc when
            goal reached
        """
        self.NOISE = noise
        self.infinite = infinite
        self.configuration = CFGS[configuration]
        self.viewer = None
        self.episodeCap = episodeCap
        self.actions_num = 5
        self.actions = [
            PinballModel.ACC_X,
            PinballModel.DEC_Y,
            PinballModel.DEC_X,
            PinballModel.ACC_Y,
            PinballModel.ACC_NONE]
        self.statespace_limits = np.array(
            [[0.0, 1.0], [0.0, 1.0], [-2.0, 2.0], [-2.0, 2.0]])
        self.continuous_dims = [4]
        #super(Pinball, self).__init__()


        self.observation_space = spaces.Box(self.statespace_limits[:,0],
                                            self.statespace_limits[:,1])
        self.action_space = spaces.Discrete(5)

        self._seed()

        self.environment = PinballModel(
            self.configuration,
            random_state=self.random_state)

        self.reset()


    def _seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #add obstacles
            for obs in self.environment.obstacles:
                points = list(map(lambda p: (p[0]*screen_width,p[1]*screen_height),
                             obs.points))
                obj = rendering.make_polyline(points+[points[0]])
                self.viewer.add_geom(obj)
            #add target
            target_rad = self.environment.target_rad * screen_height
            target = rendering.make_circle(target_rad)
            target.set_color(0,0,1)
            self.targettrans = rendering.Transform()
            target.add_attr(self.targettrans)
            self.targettrans.set_translation(
                self.environment.target_pos[0]*screen_width,
                self.environment.target_pos[1]*screen_height)
            self.viewer.add_geom(target)
            #add ball
            ball_rad = self.environment.ball.radius * screen_height
            ball = rendering.make_circle(ball_rad)
            ball.set_color(1,0,0)
            self.balltrans = rendering.Transform()
            ball.add_attr(self.balltrans)
            self.viewer.add_geom(ball)
        #update ball location
        self.balltrans.set_translation(
            self.state[0]*screen_width,
            self.state[1]*screen_height)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _get_ob(self):
        s = self.state
        return np.array(s)

    def random_point(self,rect):
        '''select random valid point within rectangular region'''
        x1,y1,x2,y2 = rect
        p = np.array((x1,y1))+np.random.rand(2)*(x2-x1,y2-y1)
        while not self._is_valid(p):
            # will run forever if no valid points can be found
            p = np.array((x1,y1))+np.random.rand(2)*(x2-x1,y2-y1)
        return p

    def _is_valid(self, p):
        ''' check that point doesn't interesct with pinball obstacles'''
        if not util.in_rect(p,(0.,0.,1.,1.)):
            return False
        for obs in self.environment.obstacles:
            if util.intersects(p, obs):
                return False
        return True

    def step(self, a):
        s = self.state
        [self.environment.ball.position[0],
         self.environment.ball.position[1],
         self.environment.ball.xdot,
         self.environment.ball.ydot] = s
        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.get_act())
        reward = self.environment.take_action(a)
        self.environment._check_bounds()
        if self.infinite and self._terminal():
            while self._terminal(): # make sure we don't reset to terminal state
                #get random valid position
                pos = self._random_point((0.,0.,1.,1.))
                #reset environment to pos, velocity 0
                self.environment.ball.position[0] = pos[0]
                self.environment.ball.position[1] = pos[1]
                self.environment.ball.xdot = 0.0
                self.environment.ball.ydot = 0.0
        state = np.array(self.environment.get_state())
        self.state = state.copy()
        return self._get_ob(), reward, self._terminal(), {}

    def reset(self):
        self.environment.ball.position[0], self.environment.ball.position[
            1] = self.environment.start_pos
        self.environment.ball.xdot, self.environment.ball.ydot = 0.0, 0.0
        self.state = np.array(
            [self.environment.ball.position[0],
             self.environment.ball.position[1],
             self.environment.ball.xdot,
             self.environment.ball.ydot])
        return self._get_ob()

    def get_act(self, s=0):
        return np.array(self.actions)

    def _terminal(self):
        return self.environment.episode_ended()


class BallModel(object):

    """ This class maintains the state of the ball
    in the pinball domain. It takes care of moving
    it according to the current velocity and drag coefficient.

    """
    DRAG = 0.995

    def __init__(self, start_position, radius):
        """
        :param start_position: The initial position
        :type start_position: float
        :param radius: The ball radius
        :type radius: float
        """
        self.position = start_position
        self.radius = radius
        self.xdot = 0.0
        self.ydot = 0.0

    def add_impulse(self, delta_xdot, delta_ydot):
        """ Change the momentum of the ball
        :param delta_xdot: The change in velocity in the x direction
        :type delta_xdot: float
        :param delta_ydot: The change in velocity in the y direction
        :type delta_ydot: float
        """
        self.xdot += delta_xdot / 5.0
        self.ydot += delta_ydot / 5.0
        self.xdot = self._clip(self.xdot)
        self.ydot = self._clip(self.ydot)

    def add_drag(self):
        """ Add a fixed amount of drag to the current velocity """
        self.xdot *= self.DRAG
        self.ydot *= self.DRAG

    def step(self):
        """ Move the ball by one increment """
        self.position[0] += self.xdot * self.radius / 20.0
        self.position[1] += self.ydot * self.radius / 20.0

    def _clip(self, val, low=-2, high=2):
        """ Clip a value in a given range """
        if val > high:
            val = high
        if val < low:
            val = low
        return val


class PinballObstacle(object):

    """ This class represents a single polygon obstacle in the
    pinball domain and detects when a :class:`BallModel` hits it.

    When a collision is detected, it also provides a way to
    compute the appropriate effect to apply on the ball.
    """

    def __init__(self, points):
        """
        :param points: A list of points defining the polygon
        :type points: list of lists
        """
        self.points = [(x,y) for x,y in points]
        self.min_x = min(self.points, key=lambda pt: pt[0])[0]
        self.max_x = max(self.points, key=lambda pt: pt[0])[0]
        self.min_y = min(self.points, key=lambda pt: pt[1])[1]
        self.max_y = max(self.points, key=lambda pt: pt[1])[1]

        self._double_collision = False
        self._intercept = None

    def collision(self, ball):
        """ Determines if the ball hits this obstacle

    :param ball: An instance of :class:`BallModel`
    :type ball: :class:`BallModel`
        """
        self._double_collision = False

        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        a, b = tee(np.vstack([np.array(self.points), self.points[0]]))
        next(b, None)
        intercept_found = False
        for pt_pair in zip(a, b):
            if self._intercept_edge(pt_pair, ball):
                if intercept_found:
                    # Ball has hit a corner
                    self._intercept = self._select_edge(
                        pt_pair,
                        self._intercept,
                        ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    intercept_found = True

        return intercept_found

    def collision_effect(self, ball):
        """ Based of the collision detection result triggered
    in :func:`PinballObstacle.collision`, compute the
        change in velocity.

    :param ball: An instance of :class:`BallModel`
    :type ball: :class:`BallModel`

        """
        if self._double_collision:
            return [-ball.xdot, -ball.ydot]

        # Normalize direction
        obstacle_vector = self._intercept[1] - self._intercept[0]
        if obstacle_vector[0] < 0:
            obstacle_vector = self._intercept[0] - self._intercept[1]

        velocity_vector = np.array([ball.xdot, ball.ydot])
        theta = self._angle(velocity_vector, obstacle_vector) - np.pi
        if theta < 0:
            theta += 2 * np.pi

        intercept_theta = self._angle([-1, 0], obstacle_vector)
        theta += intercept_theta

        if theta > 2 * np.pi:
            theta -= 2 * np.pi

        velocity = np.linalg.norm([ball.xdot, ball.ydot])

        return [velocity * np.cos(theta), velocity * np.sin(theta)]

    def _select_edge(self, intersect1, intersect2, ball):
        """ If the ball hits a corner, select one of two edges.

    :param intersect1: A pair of points defining an edge of the polygon
    :type intersect1: list of lists
    :param intersect2: A pair of points defining an edge of the polygon
    :type intersect2: list of lists
    :returns: The edge with the smallest angle with the velocity vector
    :rtype: list of lists

        """
        velocity = np.array([ball.xdot, ball.ydot])
        obstacle_vector1 = intersect1[1] - intersect1[0]
        obstacle_vector2 = intersect2[1] - intersect2[0]

        angle1 = self._angle(velocity, obstacle_vector1)
        if angle1 > np.pi:
            angle1 -= np.pi

        angle2 = self._angle(velocity, obstacle_vector2)
        if angle1 > np.pi:
            angle2 -= np.pi

        if np.abs(angle1 - (np.pi / 2.0)) < np.abs(angle2 - (np.pi / 2.0)):
            return intersect1
        return intersect2

    def _angle(self, v1, v2):
        """ Compute the angle difference between two vectors

    :param v1: The x,y coordinates of the vector
    :type: v1: list
    :param v2: The x,y coordinates of the vector
    :type: v2: list
    :rtype: float

    """
        angle_diff = np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1])
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        return angle_diff

    def _intercept_edge(self, pt_pair, ball):
        """ Compute the projection on and edge and find out
    if it intercepts with the ball.
    :param pt_pair: The pair of points defining an edge
    :type pt_pair: list of lists
    :param ball: An instance of :class:`BallModel`
    :type ball: :class:`BallModel`
    :returns: True if the ball has hit an edge of the polygon
    :rtype: bool

        """
        # Find the projection on an edge
        obstacle_edge = pt_pair[1] - pt_pair[0]
        difference = np.array(ball.position) - pt_pair[0]

        scalar_proj = difference.dot(
            obstacle_edge) / obstacle_edge.dot(obstacle_edge)
        if scalar_proj > 1.0:
            scalar_proj = 1.0
        elif scalar_proj < 0.0:
            scalar_proj = 0.0

        # Compute the distance to the closest point
        closest_pt = pt_pair[0] + obstacle_edge * scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance <= ball.radius * ball.radius:
            # A collision only if the ball is not already moving away
            velocity = np.array([ball.xdot, ball.ydot])
            ball_to_obstacle = closest_pt - ball.position

            angle = self._angle(ball_to_obstacle, velocity)
            if angle > np.pi:
                angle = 2 * np.pi - angle

            if angle > np.pi / 1.99:
                return False

            return True
        else:
            return False


class PinballModel(object):

    """ This class is a self-contained model of the pinball
    domain for reinforcement learning.

    It can be used either over RL-Glue through the :class:`PinballRLGlue`
    adapter or interactively with :class:`PinballView`.

    """
    ACC_X = 0
    ACC_Y = 1
    DEC_X = 2
    DEC_Y = 3
    ACC_NONE = 4

    STEP_PENALTY = -1
    THRUST_PENALTY = -5
    END_EPISODE = 10000

    def __init__(self, configuration, random_state=np.random.RandomState()):
        """ Read a configuration file for Pinball and draw the domain to screen

    :param configuration: a configuration file containing the polygons,
        source(s) and target location.
    :type configuration: str

        """

        self.random_state = random_state
        self.action_effects = {self.ACC_X: (1, 0), self.ACC_Y: (
            0, 1), self.DEC_X: (-1, 0), self.DEC_Y: (0, -1), self.ACC_NONE: (0, 0)}

        # Set up the environment according to the configuration
        self.obstacles = []
        self.target_pos = []
        self.target_rad = 0.01

        ball_rad = 0.01
        start_pos = []
        fp=StringIO.StringIO(configuration)
        for line in fp.readlines():
            tokens = line.strip().split()
            if not len(tokens):
                continue
            elif tokens[0] == 'polygon':
                self.obstacles.append(
                    PinballObstacle(zip(*[iter(map(float, tokens[1:]))] * 2)))
            elif tokens[0] == 'target':
                self.target_pos = [float(tokens[1]), float(tokens[2])]
                self.target_rad = float(tokens[3])
            elif tokens[0] == 'start':
                start_pos = zip(*[iter(map(float, tokens[1:]))] * 2)
            elif tokens[0] == 'ball':
                ball_rad = float(tokens[1])
        start_pos = [(x,y) for x,y in start_pos]
        self.start_pos = start_pos[0]
        a = self.random_state.randint(len(start_pos))
        self.ball = BallModel(list(start_pos[a]), ball_rad)

    def get_state(self):
        """ Access the current 4-dimensional state vector

        :returns: a list containing the x position, y position, xdot, ydot
        :rtype: list

        """
        return (
            [self.ball.position[0],
             self.ball.position[1],
             self.ball.xdot,
             self.ball.ydot]
        )

    def take_action(self, action):
        """ Take a step in the environment

        :param action: The action to apply over the ball
        :type action: int

        """
        for i in range(20):
            if i == 0:
                self.ball.add_impulse(*self.action_effects[action])

            self.ball.step()

            # Detect collisions
            ncollision = 0
            dxdy = np.array([0, 0])

            for obs in self.obstacles:
                if obs.collision(self.ball):
                    dxdy = dxdy + obs.collision_effect(self.ball)
                    ncollision += 1

            if ncollision == 1:
                self.ball.xdot = self.ball._clip(dxdy[0])
                self.ball.ydot = self.ball._clip(dxdy[1])
                if i == 19:
                    self.ball.step()
            elif ncollision > 1:
                self.ball.xdot = -self.ball.xdot
                self.ball.ydot = -self.ball.ydot

            if self.episode_ended():
                return self.END_EPISODE

        self.ball.add_drag()
        self._check_bounds()

        if action == self.ACC_NONE:
            return self.STEP_PENALTY

        return self.THRUST_PENALTY

    def episode_ended(self):
        """ Find out if the ball reached the target

        :returns: True if the ball reached the target position
        :rtype: bool

        """
        return (
            np.linalg.norm(np.array(self.ball.position)
                           - np.array(self.target_pos)) < self.target_rad
        )

    def _check_bounds(self):
        """ Make sure that the ball stays within the environment """
        if self.ball.position[0] > 1.0:
            self.ball.position[0] = 0.95
        if self.ball.position[0] < 0.0:
            self.ball.position[0] = 0.05
        if self.ball.position[1] > 1.0:
            self.ball.position[1] = 0.95
        if self.ball.position[1] < 0.0:
            self.ball.position[1] = 0.05
