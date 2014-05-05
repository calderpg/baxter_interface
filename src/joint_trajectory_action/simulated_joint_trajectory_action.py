# Copyright (c) 2013, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Joint Trajectory Action Server
"""
import bisect
from copy import deepcopy
import math
import time
import operator
from __builtin__ import xrange

import rospy

import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryResult,
)
from std_msgs.msg import (
    UInt16,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

import baxter_control
import baxter_dataflow
import baxter_interface

# Add XTF to provide logging support
import xtf.xtf as XTF
import deformable_astar.grc as GRC
from baxter_uncertainty.srv import *
import numpy
import random
from sensor_msgs.msg import JointState


class SimulatedJointTrajectoryActionServer(object):

    def __init__(self, limb, reconfig_server, rate=100.0, disable_preemption=True):
        self._dyn = reconfig_server
        self._ns = 'robot/limb/' + limb + '/follow_joint_trajectory'
        self._server = actionlib.SimpleActionServer(self._ns, FollowJointTrajectoryAction, execute_cb=self._grc_trajectory_action, auto_start=False)
        self._action_name = rospy.get_name()
        self._server.start()
        if limb == "left":
            self._joint_names = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
        elif limb == "right":
            self._joint_names = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"]

        # Initialize the current state
        self._current_position = []
        for idx in range(len(self._joint_names)):
            self._current_position.append(0.0)

        # Set up joint state feedback
        self._state_sub = rospy.Subscriber("joint_states", JointState, self._state_cb)

        # Allow preemption to be disabled for testing
        self._disable_preemption = disable_preemption

        # Make a parser for XTF trajectories
        self._parser = XTF.XTFParser()

        # Make a GradientRejectionController
        self._grcontroller = GRC.GradientRejectionController()

        # Set the bound used by the Gradient Rejection Controller
        self._grc_control_factor = 10.0

        # Set the gain to use when computing corrections
        self._grc_gain = 0.25

        # Set the uncertainty params for simulated actuation
        self._minimum_variance = 0.001
        self._variance_scaling = 0.001

        # Setup the fake command publisher
        self._command_pub = rospy.Publisher('joint_commands', JointState)

        # Set the amount to "overtime" each step of a trajectory - we do this to give Baxter an easier time
        self._overtime_multiplier = 1.0

        # Set the location for log files
        self._log_location = "/home/calderpg/Desktop/trajectory_examples/"

        # Set the correction mode
        self._control_mode = ComputeGradientRequest.POINT_NEIGHBORS
        #self._control_mode = ComputeGradientRequest.GRADIENT_BLACKLIST
        #self._control_mode = ComputeGradientRequest.SAMPLED_EIGHT_CONNECTED
        #self._control_mode = ComputeGradientRequest.HYBRID_FOUR_CONNECTED
        #self._control_mode = ComputeGradientRequest.FOUR_CONNECTED

        # Make a client to query the feature server
        self._feature_client = rospy.ServiceProxy("compute_baxter_cost_features", ComputeFeatures, persistent=True)

        # Make a client to query the uncertainty gradient server
        self._gradient_client = rospy.ServiceProxy("compute_baxter_cost_uncertainty_gradient", ComputeGradient, persistent=True)

        # Action Feedback/Result
        self._result = FollowJointTrajectoryResult()

        # Controller parameters from arguments and dynamic reconfigure
        self._control_rate = rate  # Hz
        self._control_joints = []
        self._pid_gains = {'kp': dict(), 'ki': dict(), 'kd': dict()}
        self._goal_time = 0.0
        self._goal_error = dict()
        self._error_threshold = dict()
        self._dflt_vel = dict()

    def _state_cb(self, msg):
        new_state = []
        for name in self._joint_names:
            new_state.append(msg.position[msg.name.index(name)])
        self._current_position = new_state

    def _call_feature_client_safe(self, req, max_tries=5):
        try:
            return self._feature_client.call(req)
        except:
            if max_tries == 0:
                raise AttributeError("Feature client cannot connect")
            self._feature_client = rospy.ServiceProxy("compute_baxter_cost_features", ComputeFeatures, persistent=True)
            return self._call_feature_client_safe(req, (max_tries - 1))

    def _call_gradient_client_safe(self, req, max_tries=5):
        try:
            return self._gradient_client.call(req)
        except:
            if max_tries == 0:
                raise AttributeError("Gradient client cannot connect")
            self._gradient_client = rospy.ServiceProxy("compute_baxter_cost_uncertainty_gradient", ComputeGradient, persistent=True)
            return self._call_gradient_client_safe(req, (max_tries - 1))

    def _get_trajectory_parameters(self, joint_names):
        self._goal_time = self._dyn.config['goal_time']
        for jnt in joint_names:
            if not jnt in self._joint_names:
                rospy.logerr(
                    "%s: Trajectory Aborted - Provided Invalid Joint Name %s" %
                    (self._action_name, jnt,))
                self._result.error_code = self._result.INVALID_JOINTS
                self._server.set_aborted(self._result)
                return
            goal_error = self._dyn.config[jnt + '_goal']
            self._goal_error[jnt] = goal_error
            trajectory_error = self._dyn.config[jnt + '_trajectory']
            self._error_threshold[jnt] = trajectory_error
            self._dflt_vel[jnt] = self._dyn.config[jnt + '_default_velocity']

    def _get_current_position(self, joint_names):
        temp_copy = deepcopy(self._current_position)
        ordered_config = []
        for joint_name in joint_names:
            local_index = self._joint_names.index(joint_name)
            ordered_config.append(temp_copy[local_index])
        return ordered_config

    def _get_current_error(self, joint_names, set_point):
        current = self._get_current_position(joint_names)
        error = map(operator.sub, set_point, current)
        return zip(joint_names, error)

    def _update_feedback(self, target_positions, joint_names, cur_time):
        feedback_msg = FollowJointTrajectoryFeedback()
        feedback_msg.header.stamp = rospy.get_rostime()
        feedback_msg.joint_names = joint_names
        feedback_msg.desired.positions = target_positions
        feedback_msg.desired.time_from_start = rospy.Duration.from_sec(cur_time)
        feedback_msg.actual.positions = self._get_current_position(joint_names)
        feedback_msg.actual.time_from_start = rospy.Duration.from_sec(cur_time)
        feedback_msg.error.positions = map(operator.sub, feedback_msg.desired.positions, feedback_msg.actual.positions)
        feedback_msg.error.time_from_start = rospy.Duration.from_sec(cur_time)
        self._server.publish_feedback(feedback_msg)
        return feedback_msg.actual.positions

    def _interpolate(self, a, b, percent):
        return a + ((b - a) * percent)

    def _interpolate_joints(self, p1, p2, percent):
        interpolated = []
        assert(len(p1) == len(p2))
        for [pp1, pp2] in zip(p1, p2):
            interpolated.append(self._interpolate(pp1, pp2, percent))
        return interpolated

    def _sample_resultant_state(self, current_configuration, target_configuration):
        # For each joint, sample a new joint value given the current state and control input (target-current)
        # Determine the control input
        control_input = [0.0 for idx in xrange(len(current_configuration))]
        for index in range(len(current_configuration)):
            control_input[index] = target_configuration[index] - current_configuration[index]
        # Sample a new configuration based on the control input
        sampled_configuration = [0.0 for idx in xrange(len(current_configuration))]
        for index in range(len(current_configuration)):
            sampled_configuration[index] = self._sample_joint(current_configuration[index], control_input[index])
        return sampled_configuration

    def _sample_joint(self, current_value, control_input):
        mean = current_value + control_input
        variance = (self._variance_scaling * control_input) + self._minimum_variance
        sigma = math.sqrt(abs(variance))
        return random.gauss(mean, sigma)

    def _execute_to_state(self, joint_names, target_configuration, exec_time):
        # Overtime the current step
        exec_time = exec_time * self._overtime_multiplier
        # Check the sanity of the commanded point
        if len(joint_names) != len(target_configuration):
            rospy.logerr("%s: Commanded point and joint names do not match - aborting" % (self._action_name,))
            self._server.set_aborted()
            return False
        # Check the sanity of the commanded execution time
        if exec_time == 0.0:
            rospy.logerr("%s: Execution time is infeasible - skipping" % (self._action_name,))
            return True
        # Debug print
        print("Going to new state " + str(target_configuration) + " for " + str(exec_time) + " seconds")
        # Now that we think the point is safe to execute, let's do it
        start_configuration = self._get_current_position(joint_names)
        # Sample the 'real' target
        real_target_configuration = self._sample_resultant_state(start_configuration, target_configuration)
        # Get the operating rate
        control_rate = rospy.Rate(self._control_rate)
        control_duration = 1.0 / self._control_rate
        # Loop until exec_time runs out
        start_time = rospy.get_time()
        elapsed_time = rospy.get_time() - start_time
        while elapsed_time <= (exec_time + 2 * control_duration):
            # Interpolate a current state from start and target configuration
            percent = elapsed_time / exec_time
            target_point = self._interpolate_joints(start_configuration, real_target_configuration, percent)
            # Execute to the current target
            self._command_to_state(joint_names, target_point)
            # Wait for the rest of the time step
            control_rate.sleep()
            # Update time
            elapsed_time = rospy.get_time() - start_time
        return True

    def _command_to_state(self, joint_names, joint_values):
        command_msg = JointState()
        command_msg.header.stamp = rospy.get_rostime()
        command_msg.name = joint_names
        command_msg.position = joint_values
        self._command_pub.publish(command_msg)

    def _get_ordered_configuration(self, joint_names, configuration):
        # Figure out left or right side
        if all("left" in jn for jn in joint_names):
            side = "left"
        elif all("right" in jn for jn in joint_names):
            side = "right"
        # build the ordered configuration
        if side == "left":
            print("Joint names [left] : " + str(joint_names))
            print("Configuration [left] : " + str(configuration))
            ordered_configuration = []
            ordered_configuration.append(configuration[joint_names.index('left_s0')])
            ordered_configuration.append(configuration[joint_names.index('left_s1')])
            ordered_configuration.append(configuration[joint_names.index('left_e0')])
            ordered_configuration.append(configuration[joint_names.index('left_e1')])
            ordered_configuration.append(configuration[joint_names.index('left_w0')])
            ordered_configuration.append(configuration[joint_names.index('left_w1')])
            ordered_configuration.append(configuration[joint_names.index('left_w2')])
            return ordered_configuration
        elif side == "right":
            print("Joint names [right] : " + str(joint_names))
            print("Configuration [right] : " + str(configuration))
            ordered_configuration = []
            ordered_configuration.append(configuration[joint_names.index('right_s0')])
            ordered_configuration.append(configuration[joint_names.index('right_s1')])
            ordered_configuration.append(configuration[joint_names.index('right_e0')])
            ordered_configuration.append(configuration[joint_names.index('right_e1')])
            ordered_configuration.append(configuration[joint_names.index('right_w0')])
            ordered_configuration.append(configuration[joint_names.index('right_w1')])
            ordered_configuration.append(configuration[joint_names.index('right_w2')])
            return ordered_configuration

    def _grc_trajectory_action(self, goal):
        joint_names = goal.trajectory.joint_names
        trajectory_points = goal.trajectory.points
        # Check to make sure the trajectory isn't empty
        if len(trajectory_points) == 0:
            rospy.logerr("%s: Provided an empty trajectory, aborting" % (self._action_name,))
            self._server.set_aborted()
            return
        # Check the length to see if GRC will ever be used
        if len(trajectory_points) == 1:
            rospy.logwarn("%s: Provided trajectory contains a single point, GRC will not be enabled" % (self._action_name,))
        if len(trajectory_points) == 2:
            rospy.logwarn("%s: Provided trajectory contains two points (start + goal), GRC will not be enabled" % (self._action_name))
        # Check to see if the all time_from_start values are non-zero
        # unlike Rethink's code, we will consider this an error that
        # that will not be executed
        zero_time_points = 0
        for pt in trajectory_points:
            if pt.time_from_start.to_sec() == 0.0:
                zero_time_points += 1
        if zero_time_points > 1:
            rospy.logerr("%s: Provided trajectory has invalid time_from_start values, aborting" % (self._action_name,))
            self._server.set_aborted()
            return
        elif zero_time_points == 1 and trajectory_points[0].time_from_start.to_sec() != 0.0:
            rospy.logerr("%s: Provided trajectory has invalid time_from_start values, aborting" % (self._action_name,))
            self._server.set_aborted()
            return
        # Now that we think the trajectory is valid
        rospy.loginfo("%s: Executing requested joint trajectory in GRC control mode" % (self._action_name,))
        # Make an XTF trajectory to process the execution
        if all("left" in jn for jn in joint_names):
            side = "left"
        elif all("right" in jn for jn in joint_names):
            side = "right"
        else:
            rospy.logerr("%s: Unable to identify which arm is being used - aborting" % (self._action_name,))
            self._server.set_aborted()
            return
        trajectory_name = "baxter_trajectory_" + side + time.strftime("_%d-%m-%Y_%H-%M-%S_executed.xtf")
        current_xtf = XTF.XTFTrajectory(trajectory_name, "recorded", "timed", "joint", "baxter", "baxter_FJTA", None, None, joint_names, [], [])
        # Convert the entire trajectory into an XTF trajectory
        rospy.loginfo("Converting trajectory to XTF and computing features...");
        sequence = 0
        for point in trajectory_points:
            configuration = self._get_ordered_configuration(joint_names, point.positions)
            new_xtf_state = XTF.XTFState(configuration, [], [], [], [], [], sequence, point.time_from_start.to_sec())
            # Get features for the current state
            if side == "left":
                req = ComputeFeaturesRequest()
                req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
                req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                #req.FeatureOption = ComputeFeaturesRequest.COST_AND_BRITTLENESS
                req.LeftArmConfiguration = new_xtf_state.position_desired
                req.GradientMultiplier = 0.1
                res = self._call_feature_client_safe(req)
            elif side == "right":
                req = ComputeFeaturesRequest()
                req.ArmOption = ComputeFeaturesRequest.RIGHT_ARM_ONLY
                req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                #req.FeatureOption = ComputeFeaturesRequest.COST_AND_BRITTLENESS
                req.RightArmConfiguration = new_xtf_state.position_desired
                req.GradientMultiplier = 0.1
                res = self._call_feature_client_safe(req)
            # Save the computed features
            if side == "left":
                new_xtf_state.extras["state_cost"] = res.LeftArmCost
                #new_xtf_state.extras["state_brittleness"] = res.LeftArmBrittleness
            elif side == "right":
                new_xtf_state.extras["state_cost"] = res.RightArmCost
                #new_xtf_state.extras["state_brittleness"] = res.RightArmBrittleness
            # Save the point
            current_xtf.trajectory.append(new_xtf_state)
            sequence += 1
        rospy.loginfo("...conversion finished")
        rospy.loginfo("Preprocessing trajectory...")
        # Preprocess the trajectory to identify GRC use
        for state in current_xtf.trajectory:
            # Dummy preprocess code
            state.extras["use_grc"] = True
        rospy.loginfo("...preprocessing complete")
        # Save a copy before execution
        self._parser.ExportTraj(current_xtf, self._log_location + "preliminary_" + current_xtf.uid)
        # Get ready to execute the trajectory for real
        # Load parameters for trajectory
        self._get_trajectory_parameters(joint_names)
        # Reset and send the first feedback message
        start_state = current_xtf.trajectory[0]
        self._update_feedback(deepcopy(start_state.position_desired), joint_names, rospy.get_time())
        # Check the starting time of the trajectory - if none provided, use current time
        start_time = goal.trajectory.header.stamp.to_sec()
        if start_time == 0.0:
            start_time = rospy.get_time()
        # Wait for the start time
        baxter_dataflow.wait_for(lambda: rospy.get_time() >= start_time, timeout=float('inf'))
        ################################################################################################################
        # Actually execute the trajectory
        ################################################################################################################
        # Execute to the first state of the trajectory
        target_point = start_state.position_desired
        exec_time = rospy.Duration(start_state.secs, start_state.nsecs).to_sec()
        if side == "left":
            ordered_joint_names = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
        elif side == "right":
            ordered_joint_names = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"]
        continue_execution = self._execute_to_state(ordered_joint_names, target_point, exec_time)
        if not continue_execution:
            rospy.logerr("%s: Start state execution failed, aborting" % (self._action_name,))
        # Update feedback and get the current state
        current_point = self._update_feedback(target_point, joint_names, rospy.get_time())
        current_point = self._get_ordered_configuration(joint_names, current_point)
        # Get the cost of the current (real) point
        if side == "left":
            req = ComputeFeaturesRequest()
            req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
            req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
            req.LeftArmConfiguration = current_point
            req.GradientMultiplier = 0.1
            res = self._call_feature_client_safe(req)
        elif side == "right":
            req = ComputeFeaturesRequest()
            req.ArmOption = ComputeFeaturesRequest.RIGHT_ARM_ONLY
            req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
            req.RightArmConfiguration = current_point
            req.GradientMultiplier = 0.1
            res = self._call_feature_client_safe(req)
        if side == "left":
            start_state.extras["executed_cost"] = res.LeftArmCost
        elif side == "right":
            start_state.extras["executed_cost"] = res.RightArmCost
        start_state.position_actual = current_point
        # Check to make sure execution is OK
        if continue_execution:
            # Loop through the trajectory points (point 2 to point n-1)
            for point_index in xrange(1, len(trajectory_points) - 1):
                current_state = current_xtf.trajectory[point_index]
                prev_state = current_xtf.trajectory[point_index - 1]
                # Adjust the target as needed
                if "use_grc" in current_state.extras.keys() and current_state.extras["use_grc"]:
                    # Get the next target position
                    next_target_position = current_xtf.trajectory[point_index + 1].position_desired
                    # Get the current real position
                    current_real_position = prev_state.position_actual
                    # Get the current ideal position
                    current_ideal_position = prev_state.position_desired
                    # Get the current target position
                    current_target_position = current_state.position_desired
                    # Get the current cost+uncertainty gradient
                    if side == "left":
                        req = ComputeGradientRequest()
                        req.ArmGradientOption = ComputeGradientRequest.LEFT_ARM_ONLY
                        req.ControlGenerationMode = self._control_mode
                        req.MaxControlsToCheck = 26
                        req.ExpectedCostSamples = 100
                        req.LeftArmConfiguration = current_real_position
                        req.GradientMultiplier = self._grc_gain
                        res = self._call_gradient_client_safe(req)
                        current_gradient = res.LeftArmGradient
                    elif side == "right":
                        req = ComputeGradientRequest()
                        req.ArmGradientOption = ComputeGradientRequest.RIGHT_ARM_ONLY
                        req.ControlGenerationMode = ComputeGradientRequest.POINT_NEIGHBORS
                        req.MaxControlsToCheck = 26
                        req.ExpectedCostSamples = 100
                        req.RightArmConfiguration = current_real_position
                        req.GradientMultiplier = self._grc_gain
                        res = self._call_gradient_client_safe(req)
                        current_gradient = res.RightArmGradient
                    # Check if we've gotten an empty gradient
                    if len(current_gradient) != 7:
                        current_gradient = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    # Make sure the current gradient is not NAN
                    if any(math.isnan(val) for val in current_gradient):
                        rospy.logwarn("%s: Current gradient is NaN, setting to zero for safety" % (self._action_name,))
                        current_gradient = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    # Actually call the GRC
                    corrected_target = self._grcontroller.compute_new_target(numpy.array(current_ideal_position), numpy.array(current_real_position), numpy.array(current_target_position), numpy.array(next_target_position), numpy.array(current_gradient), self._grc_control_factor)
                    # Make sure the corrected target is back to 'real' Python and not numpy
                    grc_corrected_target = []
                    for value in corrected_target:
                        grc_corrected_target.append(float(value))
                else:
                    # GRC is disabled, go to the original target point
                    grc_corrected_target = current_state.position_desired
                # Execute the current step
                target_point = grc_corrected_target
                exec_time = rospy.Duration(current_state.secs, current_state.nsecs).to_sec() - rospy.Duration(prev_state.secs, prev_state.nsecs).to_sec()
                continue_execution = self._execute_to_state(ordered_joint_names, target_point, exec_time)
                if not continue_execution:
                    rospy.logerr("%s: Middle state execution failed, aborting" % (self._action_name,))
                # Update feedback and get the current state
                current_point = self._update_feedback(target_point, joint_names, rospy.get_time())
                current_point = self._get_ordered_configuration(joint_names, current_point)
                # Get the cost of the current (real) point
                if side == "left":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.LeftArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    res = self._call_feature_client_safe(req)
                elif side == "right":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.RIGHT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.RightArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    res = self._call_feature_client_safe(req)
                if side == "left":
                    current_state.extras["executed_cost"] = res.LeftArmCost
                elif side == "right":
                    current_state.extras["executed_cost"] = res.RightArmCost
                current_state.position_actual = current_point
                # Stop looping if execution has failed
                if not continue_execution:
                    break
            # Execute to the final state of the trajectory
            if continue_execution:
                end_state = current_xtf.trajectory[-1]
                prev_state = current_xtf.trajectory[-2]
                target_point = end_state.position_desired
                exec_time = rospy.Duration(end_state.secs, end_state.nsecs).to_sec() - rospy.Duration(prev_state.secs, prev_state.nsecs).to_sec()
                continue_execution = self._execute_to_state(ordered_joint_names, target_point, exec_time)
                if not continue_execution:
                    rospy.logerr("%s: End state execution failed, aborting" % (self._action_name,))
                # Update feedback and get the current state
                current_point = self._update_feedback(target_point, joint_names, rospy.get_time())
                current_point = self._get_ordered_configuration(joint_names, current_point)
                # Get the cost of the current (real) point
                if side == "left":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.LeftArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    res = self._call_feature_client_safe(req)
                elif side == "right":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.RIGHT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.RightArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    res = self._call_feature_client_safe(req)
                if side == "left":
                    end_state.extras["executed_cost"] = res.LeftArmCost
                elif side == "right":
                    end_state.extras["executed_cost"] = res.RightArmCost
                end_state.position_actual = current_point
            else:
                # Execution has been halted due to an error
                rospy.logerr("%s: Aborting further execution" % (self._action_name,))
        else:
            # Execution has been halted due to an error
            rospy.logerr("%s: Aborting further execution" % (self._action_name,))
        ################################################################################################################
        # Trajectory execution is now complete - check status and finish
        ################################################################################################################
        # Now that we have run out of time, check to see if we reached the goal
        goal_met = continue_execution
        final_error = self._get_current_error(joint_names, trajectory_points[-1].positions)
        for [name, error] in final_error:
            if self._goal_error[name] > 0.0:
                if abs(error) > abs(self._goal_error[name]):
                    goal_met = False
                    rospy.logerr("%s: Goal tolerance for joint %s violated - error is %f, tolerance is %f" % (self._action_name, name, error, self._goal_error[name]))
            else:
                rospy.logwarn("No goal tolerance for joint %s, skipping tolerance check" % (name,))
        if goal_met:
            rospy.loginfo("%s: Goal execution complete" % (self._action_name,))
            self._result.error_code = self._result.SUCCESSFUL
            self._server.set_succeeded(self._result)
            current_xtf.tags.append("successful")
        elif continue_execution:
            rospy.logerr("%s: Aborting due to goal tolerance violation" % (self._action_name,))
            self._result.error_code = self._result.GOAL_TOLERANCE_VIOLATED
            self._server.set_aborted(self._result)
            current_xtf.tags.append("goal_tolerance_violated")
        else:
            rospy.logerr("%s: Already aborted" % (self._action_name,))
        # Write the XTF trajectory to disk
        self._parser.ExportTraj(current_xtf, self._log_location + current_xtf.uid)
        rospy.loginfo("%s: FollowJointTrajectory action call complete, trajectory execution logged to file %s" % (self._action_name, current_xtf.uid,))
