import os
import torch
import numpy as np
import pandas as pd
import cv2
import pydicom

if __name__ == "__main__":
    dcm1 = pydicom.read_file('/home/yzy/Desktop/alldcm/1/MR.1.2.156.112605.14038000790728.20160514050558.4.12164.1.dcm')
    dcm2 = pydicom.read_file('/home/yzy/Desktop/alldcm/2/MR.1.2.156.112605.14038000790728.20160514053228.4.11660.1.dcm')
    dcm3 = pydicom.read_file('/home/yzy/Desktop/alldcm/23/MR.1.2.156.112605.14038000790728.20160512034246.4.13080.1.dcm')
    dcm4 = pydicom.read_file('/home/yzy/Desktop/alldcm/65/MR.1.2.156.112605.14038000790728.20160506122540.4.18624.1.dcm')
    dcm5 = pydicom.read_file('/home/yzy/Desktop/alldcm/131/MR.1.2.156.112605.14038000790728.20160426055113.4.17564.1.dcm')
    dcm6 = pydicom.read_file('/home/yzy/Desktop/alldcm/4/MR.1.3.46.670589.11.42457.5.0.4828.2016051416573946524.dcm')
    dcm7 = pydicom.read_file('/home/yzy/Desktop/alldcm/7/MR.1.3.46.670589.11.42457.5.0.4828.2016051418124269919.dcm')
    dcm8 = pydicom.read_file('/home/yzy/Desktop/alldcm/38/MR.1.3.46.670589.11.42457.5.0.4828.2016051018540397591.dcm')
    dcm9 = pydicom.read_file('/home/yzy/Desktop/alldcm/87/MR.1.3.46.670589.11.42457.5.0.5012.2016050319455821886.dcm')
    dcm10 = pydicom.read_file('/home/yzy/Desktop/alldcm/133/MR.1.3.46.670589.11.42457.5.0.5968.2016042509130516530.dcm')
    '''
['AccessionNumber', 'AcquisitionDate', 'AcquisitionDateTime', 'AcquisitionDuration', 'AcquisitionTime', 'BitsAllocated', 'BitsStored', 'BodyPartExamined', 'BurnedInAnnotation', 'Columns', 'ComplexImageComponent', 'ContentDate', 'ContentTime', 'DeviceSerialNumber', 'EchoNumbers', 'EchoTime', 'EchoTrainLength', 'FillerOrderNumberImagingServiceRequest', 'FlipAngle', 'FrameOfReferenceUID', 'HighBit', 'ImageComments', 'ImageOrientationPatient', 'ImagePositionPatient', 'ImageType', 'ImagedNucleus', 'ImagingFrequency', 'InPlanePhaseEncodingDirection', 'InstanceCreationDate', 'InstanceCreationTime', 'InstanceNumber', 'InstitutionAddress', 'InstitutionName', 'IssuerOfPatientID', 'LargestImagePixelValue', 'Laterality', 'LossyImageCompression', 'MRAcquisitionType', 'MagneticFieldStrength', 'Manufacturer', 'ManufacturerModelName', 'Modality', 'NumberOfAverages', 'NumberOfSlices', 'OperatorsName', 'PatientAddress', 'PatientAge', 'PatientBirthDate', 'PatientComments', 'PatientID', 'PatientName', 'PatientPosition', 'PatientSex', 'PatientSize', 'PatientWeight', 'PerformingPhysicianName', 'PhotometricInterpretation', 'PixelBandwidth', 'PixelData', 'PixelRepresentation', 'PixelSpacing', 'PositionReferenceIndicator', 'ProcedureCodeSequence', 'ProtocolName', 'ReferringPhysicianName', 'RepetitionTime', 'RequestAttributesSequence', 'RequestingPhysician', 'RequestingService', 'Rows', 'SAR', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'ScanOptions', 'ScanningSequence', 'SequenceName', 'SequenceVariant', 'SeriesDate', 'SeriesDescription', 'SeriesInstanceUID', 'SeriesNumber', 'SeriesTime', 'SliceLocation', 'SliceThickness', 'SmallestImagePixelValue', 'SoftwareVersions', 'SpacingBetweenSlices', 'SpecificCharacterSet', 'StationName', 'StudyDate', 'StudyDescription', 'StudyID', 'StudyInstanceUID', 'StudyTime', 'TransmitCoilName', 'TriggerSourceOrType', 'WindowCenter', 'WindowWidth', 'dBdt']
    '''
    '''
['AccessionNumber', 'AcquisitionDate', 'AcquisitionDuration', 'AcquisitionMatrix', 'AcquisitionNumber', 'AcquisitionTime', 'AdditionalPatientHistory', 'AdmittingDiagnosesDescription', 'Allergies', 'BitsAllocated', 'BitsStored', 'BodyPartExamined', 'CodeMeaning', 'CodeValue', 'CodingSchemeDesignator', 'Columns', 'CommentsOnThePerformedProcedureStep', 'ContentDate', 'ContentTime', 'ConversionType', 'DeviceSerialNumber', 'DiffusionBValue', 'DiffusionGradientOrientation', 'DigitalImageFormatAcquired', 'EchoNumbers', 'EchoTime', 'EchoTrainLength', 'EthnicGroup', 'FillerOrderNumberImagingServiceRequest', 'FlipAngle', 'FrameOfReferenceUID', 'HeartRate', 'HighBit', 'HighRRValue', 'ImageOrientationPatient', 'ImagePositionPatient', 'ImageType', 'ImagedNucleus', 'ImagingFrequency', 'ImagingServiceRequestComments', 'InPlanePhaseEncodingDirection', 'InstanceCreationDate', 'InstanceCreationTime', 'InstanceCreatorUID', 'InstanceNumber', 'InstitutionAddress', 'InstitutionName', 'InstitutionalDepartmentName', 'IntervalsAcquired', 'IntervalsRejected', 'IssueDateOfImagingServiceRequest', 'IssueTimeOfImagingServiceRequest', 'IssuerOfPatientID', 'Laterality', 'LowRRValue', 'MRAcquisitionType', 'MagneticFieldStrength', 'Manufacturer', 'ManufacturerModelName', 'MedicalAlerts', 'Modality', 'NumberOfAverages', 'NumberOfPhaseEncodingSteps', 'NumberOfTemporalPositions', 'Occupation', 'OperatorsName', 'OrderCallbackPhoneNumber', 'OrderEntererLocation', 'PatientAddress', 'PatientAge', 'PatientBirthDate', 'PatientComments', 'PatientID', 'PatientName', 'PatientPosition', 'PatientSex', 'PatientState', 'PatientTransportArrangements', 'PatientWeight', 'PercentPhaseFieldOfView', 'PercentSampling', 'PerformedLocation', 'PerformedProcedureStepDescription', 'PerformedProcedureStepEndDate', 'PerformedProcedureStepEndTime', 'PerformedProcedureStepID', 'PerformedProcedureStepStartDate', 'PerformedProcedureStepStartTime', 'PerformedProcedureStepStatus', 'PerformedProcedureTypeDescription', 'PerformedProtocolCodeSequence', 'PerformedStationAETitle', 'PerformedStationName', 'PerformingPhysicianName', 'PhotometricInterpretation', 'PixelBandwidth', 'PixelData', 'PixelRepresentation', 'PixelSpacing', 'PositionReferenceIndicator', 'PregnancyStatus', 'PresentationLUTShape', 'ProcedureCodeSequence', 'ProtocolName', 'ReasonForTheImagingServiceRequest', 'ReasonForTheRequestedProcedure', 'ReceiveCoilName', 'ReconstructionDiameter', 'ReferencedPerformedProcedureStepSequence', 'ReferringPhysicianName', 'RepetitionTime', 'RequestAttributesSequence', 'RequestedContrastAgent', 'RequestedProcedureComments', 'RequestedProcedureDescription', 'RequestedProcedureLocation', 'RequestedProcedurePriority', 'RequestingPhysician', 'RequestingService', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 'SAR', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'ScanOptions', 'ScanningSequence', 'ScheduledPerformingPhysicianName', 'SecondaryCaptureDeviceID', 'SecondaryCaptureDeviceManufacturer', 'SecondaryCaptureDeviceManufacturerModelName', 'SecondaryCaptureDeviceSoftwareVersions', 'SequenceVariant', 'SeriesDate', 'SeriesDescription', 'SeriesInstanceUID', 'SeriesNumber', 'SeriesTime', 'SliceLocation', 'SliceThickness', 'SoftwareVersions', 'SpacingBetweenSlices', 'SpecialNeeds', 'SpecificCharacterSet', 'StationName', 'StudyComments', 'StudyDate', 'StudyDescription', 'StudyID', 'StudyInstanceUID', 'StudyTime', 'TemporalPositionIdentifier', 'VideoImageFormatAcquired', 'WindowCenter', 'WindowWidth', 'dBdt']
    '''
    attr = 'SeriesDescription'
    print(dcm1[attr])
    print(dcm2[attr])
    print(dcm3[attr])
    print(dcm4[attr])
    print(dcm5[attr])
    print(dcm6[attr])
    print(dcm7[attr])
    print(dcm8[attr])
    print(dcm9[attr])
    print(dcm10[attr])








