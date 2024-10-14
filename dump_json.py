import json


doc = {
    "doc_name":"An Introduction to Einstein's Theory of Relativity",
    "source":"https://www.space.com/17661-theory-general-relativity.html",
    "author":"By Nola Taylor Redd",
    "date":"March 14, 2018",
    "main":"""Relativity, introduced by Albert Einstein, revolutionized our understanding of space, time, and gravity. It consists of two main theories: Special Relativity (1905) and General Relativity (1915).

Special Relativity
Special Relativity focuses on objects moving at constant speeds, particularly at or near the speed of light. Einstein proposed two key principles:

The speed of light is constant for all observers, regardless of their motion relative to the light source.
The laws of physics are the same in all inertial frames of reference (i.e., frames moving at constant velocities relative to each other).
These principles lead to some surprising consequences:

Time dilation: Time moves slower for an object in motion compared to one at rest. For example, astronauts traveling near light speed would age slower than people on Earth.
Length contraction: Objects appear shorter in the direction of motion as they approach light speed.
Mass-energy equivalence: The famous equation E = mc² shows that energy (E) and mass (m) are interchangeable, meaning mass can be converted into energy and vice versa.
General Relativity
General Relativity extends these ideas to include gravity. Einstein described gravity not as a force but as the curvature of spacetime caused by massive objects. Large masses like stars and planets bend the fabric of spacetime, and this curvature tells objects how to move. This explains phenomena like the bending of light around stars (gravitational lensing) and the slowing of time near massive objects (gravitational time dilation).

Einstein’s theories of relativity have been confirmed by numerous experiments, from the precise predictions of planetary orbits to the detection of gravitational waves. These theories form the foundation of modern physics, shaping our understanding of the universe on both cosmic and quantum scales."""
}

with open("/home/dfoadmin/boqu/insightful_summarization/input_files/document.json", 'w') as f:
    f.write(json.dumps(doc, indent=2))