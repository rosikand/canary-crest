"""
File: linear_methods.py  
-------------------
This file contains the functions that perform the linear registrations. 
Each function corresponds (by name) to a registration method described 
in the report (cr_rX). Each function also contains a header comment which 
describes the methods used. The optimizer, interpolator, and similarity 
metric are held constant. The difference lies in the transformation 
type used which is denoted in the header comment. 


Each function takes in file paths for the following (in order):
- fixed segmentation 
- moving segmentation 
- fixed scan
- moving scan
- integer index (for moved file naming purposes)
and returns:
- Dice coefficient 
- Hausdorff distance
"""


import SimpleITK as sitk
import os


def cr_r1(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 1 (CR-R1):
    - Transformation type: AffineTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.AffineTransform(fixed_gland_float.GetDimension()))
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r1/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r1/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R1, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R1, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 


def cr_r2(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 2 (CR-R2):
    - Transformation type: VersorTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.VersorTransform())
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r2/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r2/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R2, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R2, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 


def cr_r3(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 3 (CR-R3):
    - Transformation type: VersorRigid3DTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.VersorRigid3DTransform())
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r3/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r3/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R3, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R3, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 


def cr_r4(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 4 (CR-R4):
    - Transformation type: Euler3DTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.Euler3DTransform())
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r4/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r4/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R4, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R4, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 


def cr_r5(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 5 (CR-R5):
    - Transformation type: Similarity3DTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.Similarity3DTransform())
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r5/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r5/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R5, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R5, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 



def cr_r6(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 6 (CR-R6):
    - Transformation type: ScaleVersor3DTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.ScaleVersor3DTransform())
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r6/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r6/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R6, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R6, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 


def cr_r7(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 7 (CR-R7):
    - Transformation type: ScaleTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.ScaleTransform(fixed_gland_float.GetDimension()))
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r7/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r7/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R7, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R7, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 


def cr_r8(fixed_gland_path, moving_gland_path, fixed_image_path, moving_image_path, idx):
    """
    Classical registration rigid method 8 (CR-R8):
    - Transformation type: ScaleSkewVersor3DTransform (rigid)
    - Metric: Mattes mutual information
    - Interpolator: linear
    - Optimizer: gradient descent
    """

    # I. Inititialize the origins for the labels (glands) 
    # ---------------------------------------------------
    # Read Fixed (T2) Gland 
    fixed_gland =  sitk.ReadImage(fixed_gland_path, sitk.sitkInt16)
    fixed_gland_float = sitk.Cast(fixed_gland, sitk.sitkFloat32)
    
    # Read Moving (DWI) Glands
    moving_gland = sitk.ReadImage(moving_gland_path, sitk.sitkInt16)
    moving_gland_float = sitk.Cast(moving_gland, sitk.sitkFloat32)
    
    # Initialize gland origin (center)  
    initial_gland_transform = sitk.CenteredTransformInitializer(fixed_gland_float, 
                                                      moving_gland_float, 
                                                      sitk.ScaleSkewVersor3DTransform())
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
    R.SetMetricAsMattesMutualInformation()
    R.MetricUseFixedImageGradientFilterOff()
    #R.SetMetricFixedMask(fixed_gland) # use segmentation mask
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 
                                              100, 
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
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
    sitk.WriteImage(final_transform_moved_image, "outputs/cr-r8/" + "Bx_" + str(idx) + "_" 
                    + "moved_scan" + ".nii.gz")

    # moved gland 
    sitk.WriteImage(final_transform_moved_gland, "outputs/cr-r8/" + "Bx_" + str(idx) + "_" 
                    + "moved_segmentation" + ".nii.gz")
    
    # V. Calculate metrics and return 
    # ---------------------------------------------------------------
    # Print Dice score (between fixed and moved segmentations)
    dice_filter_gland = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    dice_coefficient_value = dice_filter_gland.GetDiceCoefficient()
    print("Dice score (CR-R8, Bx_" + str(idx) + "): " + str(dice_coefficient_value))
    
    # Print Hausdorff distance (between fixed and moved segmentations) 
    hausdorff_distance_filter_gland = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter_gland.Execute(fixed_gland, final_transform_moved_gland)
    hausdorff_distance_value = hausdorff_distance_filter_gland.GetHausdorffDistance()
    print("Hausdorff distance (CR-R8, Bx_" + str(idx) + "): " + str(hausdorff_distance_value))

    print("----------------------------------------------")
    
    return dice_coefficient_value, hausdorff_distance_value 

