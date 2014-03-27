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


class JointTrajectoryActionServer(object):

    def __init__(self, limb, reconfig_server, rate=100.0):
        self._dyn = reconfig_server
        self._ns = 'robot/limb/' + limb + '/follow_joint_trajectory'
        self._server = actionlib.SimpleActionServer(self._ns, FollowJointTrajectoryAction, execute_cb=self._grc_trajectory_action, auto_start=False)
        self._action_name = rospy.get_name()
        self._server.start()
        self._limb = baxter_interface.Limb(limb)

        # Make a parser for XTF trajectories
        self._parser = XTF.XTFParser()

        # Make a GradientRejectionController
        self._grcontroller = GRC.GradientRejectionController()

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

        # Create our PID controllers
        self._pid = dict()
        for joint in self._limb.joint_names():
            self._pid[joint] = baxter_control.PID()

        # Set joint state publishing to specified control rate
        self._pub_rate = rospy.Publisher('/robot/joint_state_publish_rate', UInt16)
        self._pub_rate.publish(self._control_rate)

    def _get_trajectory_parameters(self, joint_names):
        self._goal_time = self._dyn.config['goal_time']
        for jnt in joint_names:
            if not jnt in self._limb.joint_names():
                rospy.logerr(
                    "%s: Trajectory Aborted - Provided Invalid Joint Name %s" %
                    (self._action_name, jnt,))
                self._result.error_code = self._result.INVALID_JOINTS
                self._server.set_aborted(self._result)
                return

            self._pid[jnt].set_kp(self._dyn.config[jnt + '_kp'])
            self._pid[jnt].set_ki(self._dyn.config[jnt + '_ki'])
            self._pid[jnt].set_kd(self._dyn.config[jnt + '_kd'])
            self._goal_error[jnt] = self._dyn.config[jnt + '_goal']
            self._error_threshold[jnt] = self._dyn.config[jnt + '_trajectory']
            self._dflt_vel[jnt] = self._dyn.config[jnt + '_default_velocity']
            self._pid[jnt].initialize()

    def _get_current_position(self, joint_names):
        return [self._limb.joint_angle(joint) for joint in joint_names]

    def _get_current_error(self, joint_names, set_point):
        current = self._get_current_position(joint_names)
        error = map(operator.sub, set_point, current)
        return zip(joint_names, error)

    def _update_feedback(self, cmd_point, jnt_names, cur_time):
        feedback_msg = FollowJointTrajectoryFeedback()
        feedback_msg.header.stamp = rospy.get_rostime()
        feedback_msg.joint_names = jnt_names
        feedback_msg.desired = cmd_point
        feedback_msg.desired.time_from_start = rospy.Duration.from_sec(cur_time)
        feedback_msg.actual.positions = self._get_current_position(jnt_names)
        feedback_msg.actual.time_from_start = rospy.Duration.from_sec(cur_time)
        feedback_msg.error.positions = map(operator.sub, feedback_msg.desired.positions, feedback_msg.actual.positions)
        feedback_msg.error.time_from_start = rospy.Duration.from_sec(cur_time)
        self._server.publish_feedback(feedback_msg)
        return feedback_msg.actual.positions

    def _command_stop(self, joint_names):
        velocities = [0.0] * len(joint_names)
        cmd = dict(zip(joint_names, velocities))
        self._limb.set_joint_velocities(cmd)
        self._limb.set_joint_positions(self._limb.joint_angles())

    def _command_velocities(self, joint_names, positions):
        velocities = []
        if self._server.is_preempt_requested():
            self._command_stop(joint_names)
            rospy.loginfo("%s: Trajectory Preempted" % (self._action_name,))
            self._server.set_preempted()
            return False
        deltas = self._get_current_error(joint_names, positions)
        for [name, error] in deltas:
            # Check to make sure we're following the path inside the error tolerances
            if self._error_threshold[name] > 0.0:
                if abs(error) > abs(self._error_threshold[name]):
                    self._command_stop(joint_names)
                    rospy.logerr("%s: Exceeded error threshold on joint %s: %f" % (self._action_name, name, error,))
                    self._result.error_code = self._result.PATH_TOLERANCE_VIOLATED
                    self._server.set_aborted(self._result)
                    return False
            else:
                rospy.logdebug("%s: No error threshold for joint %s" % (self._action_name, name,))
            # Since we're following the path safely, compute the new velocity command with the Rethink PID controller
            velocities.append(self._pid[name].compute_output(error))
        # Combine the joint names and velocities for the Rethink API
        cmd = dict(zip(joint_names, velocities))
        # Command the arm in velocity mode with the Rethink API
        self._limb.set_joint_velocities(cmd)
        return True

    def interpolate(self, a, b, percent):
        return a + ((b - a) * percent)

    def interpolate_joints(self, p1, p2, percent):
        interpolated = []
        assert(len(p1.positions) == len(p2.positions))
        for [pp1, pp2] in zip(p1.positions, p2.positions):
            interpolated.append(self.interpolate(pp1, pp2, percent))
        return interpolated

    def _grc_trajectory_action(self, goal):
        joint_names = goal.trajectory.joint_names
        trajectory_points = goal.trajectory.points
        # Check to make sure the trajectory isn't empty
        if len(trajectory_points) == 0:
            rospy.logerr("%s: Provided an empty trajectory, aborting" % (self._action_name,))
            self._server.set_aborted()
            return
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
        # Make an XTF trajectory to log the execution
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
        # Load parameters for trajectory
        self._get_trajectory_parameters(joint_names)
        # Reset and send the first feedback message
        start_point = JointTrajectoryPoint()
        start_point.positions = self._get_current_position(joint_names)
        self._update_feedback(deepcopy(start_point), joint_names, rospy.get_time())
        # Prepare the trajectory goal into something we can actually send to baxter
        # Compute the execution time for the trajectory - add an additional timestep to help reach the goal
        trajectory_exec_time = trajectory_points[-1].time_from_start.to_sec() + (1.0 / self._control_rate)
        # Get the operating rate
        control_rate = rospy.Rate(self._control_rate)
        # Get the target times for each point in the trajectory
        pnt_times = [pnt.time_from_start.to_sec() for pnt in trajectory_points]
        # Check the starting time of the trajectory - if none provided, use current time
        start_time = goal.trajectory.header.stamp.to_sec()
        if start_time == 0.0:
            start_time = rospy.get_time()
        # Wait for the start time
        baxter_dataflow.wait_for(lambda: rospy.get_time() >= start_time, timeout=float('inf'))
        ################################################################################################################
        # Actually execute the trajectory
        ################################################################################################################
        # Loop until end of trajectory time.  Provide a single time step
        # of the control rate past the end to ensure we get to the end.
        elapsed_time = rospy.get_time() - start_time
        sequence = 0;
        while elapsed_time < trajectory_exec_time:
            # Figure out which trajectory point we are currently executing
            idx = bisect.bisect(pnt_times, elapsed_time)
            # Figure out the start point for interpolation
            if idx == 0:
                # Interpolate from start configuration if we haven't reached the first trajectory point
                p1 = deepcopy(start_point)
            else:
                # Interpolate between the previous trajectory point and the current
                p1 = deepcopy(trajectory_points[idx - 1])
            # Figure out the end point for interpolation
            if idx < len(trajectory_points):
                p2 = trajectory_points[idx]
            else:
                # If we have reached the end of the trajectory, keep going to the end state
                p2 = p1
            # Interpolate the current target from the start and end points
            time_interval = (p2.time_from_start - p1.time_from_start).to_sec()
            if time_interval > 0.0:
                percent = (elapsed_time - p1.time_from_start.to_sec()) / time_interval
            else:
                percent = 1.0
            target_point = self.interpolate_joints(p1, p2, percent)
            # Store the target in p1 for publishing feedback later
            p1.positions = target_point
            # Command the robot to the current target
            exec_status = self._command_velocities(joint_names, target_point)
            # Check the status of the execution command - if it failed, we abort
            if not exec_status:
                rospy.logerr("%s: Execution failed, aborting" % (self._action_name,))
                break
            # Wait for the rest of the time step
            control_rate.sleep()
            # Update the elapsed time
            elapsed_time = rospy.get_time() - start_time
            # Publish feedback message
            current_point = self._update_feedback(deepcopy(p1), joint_names, elapsed_time)
            # Log the current state in the XTF trajectory
            new_state = XTF.XTFState(target_point, [], [], current_point, [], [], sequence, elapsed_time)
            current_xtf.trajectory.append(new_state)
            sequence += 1
        ################################################################################################################
        # Trajectory execution is now complete - check status and finish
        ################################################################################################################
        # Now that we have run out of time, check to see if we reached the goal
        goal_met = True
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
            self._command_stop(goal.trajectory.joint_names)
            self._result.error_code = self._result.SUCCESSFUL
            self._server.set_succeeded(self._result)
            current_xtf.tags.append("successful")
        else:
            self._command_stop(goal.trajectory.joint_names)
            rospy.logerr("%s: Aborting due to goal tolerance violation" % (self._action_name,))
            self._result.error_code = self._result.GOAL_TOLERANCE_VIOLATED
            self._server.set_aborted(self._result)
            current_xtf.tags.append("goal_tolerance_violated")
        # Write the XTF trajectory to disk
        self._parser.ExportTraj(current_xtf, current_xtf.uid)
        rospy.loginfo("%s: FollowJointTrajectory action call complete, trajectory execution logged to file %s" % (self._action_name, current_xtf.uid,))