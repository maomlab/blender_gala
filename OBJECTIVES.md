# Blender Gala
Tools to support using Blender for structural biology visualization
and exploration. Standard platforms for structural biology
visualization and exploration such as PyMOL and Mol*, allow loading,
manipulating, styling, coloring, measuring, labeling, and rendering.
Recently, the Molecular Nodes plugin for Blender support of loading and
visualizing molecules leveging the high-quality and control for
Blender's rendering. Blender Gala complements Molecular Nodes to support
functionality to support Blender to be more useful for day-to-day 
structural biology visualization tasks.

## Blender Gala aims to suppor the following tools and workflows

   1) After loading a molecule, setting up publication ready scene/view
       *) Transperent background
	   *) Reasonable resolution output resolution
	   *) Setting Cycles render engine and best practices parameters for
	      quality/speed, including GPU acceleration if available, viewport denoising, etc.
	   *) Setting good color management
       *) Set Origin to the geometry of the loaded molecule
       *) Good lighting, either 3-point studio lighting using the tri-lighting
	      plugin or HDRI environmental texture.
       *) Materials controlling roughness/ambient occlusion/sub-suface scattering
	      and different materials for different types of molecules (protein vs. ligands etc.)
	   *) Use Cryptomatte compositing to facilitate tweaking the visualizations, including
          z-depth pass to control clear visualization at specific depths of field.
		  
   2) Tools for measuring and annotating structures
       *) Finding and representing interactions, including H-bond/polar contats,
	      PLiP interactions, or custom atom-atom contacts. Computing and selecting
		  interactions similar to how they are done in PyMOL. Representations
		  using solid/dashed lines that has sensible defaults and can be easily customized.
	   *) Labeling specific atoms/residues or interactions. Potentially labelling them
	      as in-schene cards, or as a easy to compositing.
	   *) Measuring distances, angles, and dihedrals using atom selection approach
	      similar to the PyMOL measurement wizard.
	   *) Coloring proteins based data annotations with AlphaFold confidence scores
	      as a primary test case.

## Implementation
As decisions and the specification is refined, record decision and requirements in SPECIFICATION.md

Implement the package as a Blender plugin that works with Molecular Nodes

Implement robust package support including
  *) Unit test
  *) Makefile to implement package tasks
  *) Use Git for version control and use CI best practices like linting, type checking etc. 
  *) Document the package through
      - Simple and clear README.md
	  - Robust and comprehensive documentation
	  - Demonstration vignettes that demonstate end-to-end use-cases and instructions
	  - A website for the package


	
