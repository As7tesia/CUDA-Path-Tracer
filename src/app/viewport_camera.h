#pragma once

#include "scene/sceneStructs.h"

#include <glm/glm.hpp>

// The viewport camera and its navigation (README "Viewport controls"), as
// math on a pose: main.cpp reads the mouse and keys and calls these. Both
// the window and headless renders build the render camera from a pose, so a
// headless render matches the window's first frame.

// Yaw turns about world +Y: 0 looks down -Z, positive turns right, toward +X.
// Pitch tilts up from the horizon and stops just short of straight up or
// down, where a basis built against world +Y would not exist. Roll then turns
// the camera about its view axis, positive tipping its right side up. It
// comes from the scene's camera and no control changes it. The orbit pivot
// sits on the view axis, pivotDistance ahead, and travels with the camera.
struct ViewportPose
{
    glm::vec3 position;
    float yaw;
    float pitch;
    float roll;
    float pivotDistance;
};

glm::vec3 viewDirection(const ViewportPose& pose);
glm::vec3 pivotOf(const ViewportPose& pose);

// The pose that sees what cam sees: the same eye, view direction and roll,
// with cam.lookAt as the pivot. A camera looking straight up or down (a glTF
// camera can) is tilted a hair off the pole. Exactly on the pole, which way
// its image is turned is not kept.
ViewportPose poseFromCamera(const Camera& cam);

// Writes pose into cam: the position, the basis generateRayFromCamera reads,
// and the pivot as lookAt. The basis is built against world +Y, then turned
// about the view by the roll.
void applyPose(const ViewportPose& pose, Camera& cam);

// Mouse navigation, for a motion of (dx, dy) pixels. mirrored is
// Camera::mirrored: a mirrored camera's screen right is its world left.

// RMB: mouse right turns toward the right of the screen, mouse up looks up.
void turn(ViewportPose& pose, double dx, double dy, bool mirrored);
// Alt + LMB: turn, then move the camera back onto the line through the
// pivot, so the pivot stays where it is on screen.
void orbit(ViewportPose& pose, double dx, double dy, bool mirrored);
// Alt + RMB: mouse right or up moves toward the pivot. The distance scales
// rather than shrinking by a fixed step, so the camera never reaches it.
void dollyToPivot(ViewportPose& pose, double dx, double dy);
// MMB, LMB + RMB, Alt + MMB: the camera moves with the mouse in its own
// plane, one pixel's width at the pivot's distance per pixel.
void pan(ViewportPose& pose, const Camera& cam, double dx, double dy);
// LMB: mouse up and down move forward and back along the level heading, as
// far as flying for GROUND_SECONDS_PER_PIXEL per pixel; left and right turn.
void moveAlongGround(ViewportPose& pose, double dx, double dy, float flySpeed, bool mirrored);

// RMB + W/S, D/A, E/Q: forward, right and up are each -1, 0 or 1, the keys
// held along the camera's view, its right and world +Y. Moves distance in
// that direction; false when no key is held and nothing moved.
bool fly(ViewportPose& pose, const Camera& cam, int forward, int right, int up, float distance);
// The wheel: distance along the view.
void moveAlongView(ViewportPose& pose, float distance);
