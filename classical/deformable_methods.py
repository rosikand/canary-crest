"""
File: deformable_methods.py  
-------------------
This file contains the functions that perform the deformable registrations. 
This includes one function: cr_d1. Note that this method is not included 
in the report. 
"""

def cr_d1(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration deformable method 1 (CR-D1):
    - Transformation type: B-Spline Transform (deformable)
    - Metric: correlation
    - Interpolator: linear
    - Optimizer: L-BFGS-B (Limited memory Broyden, Fletcher, Goldfarb, Shannon, 
    Bound Constrained)
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)

    # Deformation transform
    transformDomainMeshSize = [8] * moving_gland.GetDimension()
    initial_gland_transform = sitk.BSplineTransformInitializer(fixed_gland_float,
                                      transformDomainMeshSize)
    
    initial_transformed_gland_float=sitk.Resample(moving_gland_float, fixed_gland_float, initial_gland_transform)
    
    # Update gland 
    fixed_gland=sitk.Cast(fixed_gland_float, sitk.sitkInt16)
    moving_gland=sitk.Cast(moving_gland_float, sitk.sitkInt16)
    
    # Moved gland 
    moved_gland=sitk.Cast(initial_transformed_gland_float, sitk.sitkInt16)
    
    
    # II. Inititialize the origins for the scans (MRI sequences) 
    # ---------------------------------------------------------- 
    #  Read Fixed (T2) Image
    fixed_image =  sitk.ReadImage(fixed_image_path, sitk.sitkInt16)
    fixed_image_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_image =  sitk.ReadImage(moving_image_path, sitk.sitkInt16)
    moving_image_float = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # Initialize scan origin with the gland transform 
    initial_transformed_moving_image=sitk.Resample(moving_image_float, fixed_image_float, initial_gland_transform)
    
    
    # III. Perform the registration 
    # ---------------------------------------------------------------
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                       numberOfIterations=100,
                       maximumNumberOfCorrections=5,
                       maximumNumberOfFunctionEvaluations=1000,
                       costFunctionConvergenceFactor=1e+7)

    R.SetOptimizerScalesFromPhysicalShift()

    R.SetInitialTransform(initial_gland_transform)
    
    R.SetInterpolator(sitk.sitkLinear) 
    
    R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    R.SetSmoothingSigmasPerLevel([8, 4, 2, 1])

    # Register 
    final_transform = R.Execute(fixed_image_float, moving_image_float) 
    
    final_transform_moved_image=sitk.Resample(moving_image, fixed_image, final_transform) # scan
    final_transform_moved_gland=sitk.Resample(moving_gland, fixed_gland, final_transform) # gland

    
    # IV. Save  
    # ---------------------------------------------------------------
    # moved scan 
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-d1/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-d1/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-D1, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-D1, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 
