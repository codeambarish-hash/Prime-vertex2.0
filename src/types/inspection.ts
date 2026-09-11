/**
 * Type definitions for Vision-Based Defect Detection System
 */

export type DefectClass = 
  | 'normal'
  | 'crack'
  | 'scratch'
  | 'dent'
  | 'stain'
  | 'discoloration'
  | 'dimensional_irregularity';

export type SubstrateType = 
  | 'brushed_metal'
  | 'cast_iron'
  | 'machined_part'
  | 'ceramic_tile'
  | 'carbon_composite';

export type LightingCondition = 
  | 'uniform'
  | 'angle_glare'
  | 'low_light'
  | 'spotlight_vignette';

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface DefectRegionDetail {
  id: number;
  bbox: BoundingBox;
  areaPx: number;
  areaMm2?: number;
  widthMm?: number;
  heightMm?: number;
  centroid: { x: number; y: number };
  aspectRatio: number;
  extent: number;
}

export interface InspectionResult {
  isDefective: boolean;
  predictedClass: DefectClass;
  confidence: number;
  severity: 'None' | 'Minor' | 'Moderate' | 'Critical';
  defectPixelCount: number;
  coveragePct: number;
  bbox: BoundingBox;
  processingTimeMs: number;
  probabilities: Record<DefectClass, number>;
  regions?: DefectRegionDetail[];
  pixelToMm?: number;
  totalAreaMm2?: number;
}

export interface DatasetStatsData {
  totalImages: number;
  normalCount: number;
  defectiveCount: number;
  imageResolution: string;
  splits: {
    train: number;
    val: number;
    test: number;
  };
  classBreakdown: Record<DefectClass, { count: number; pct: number; color: string }>;
  coverageMetrics: {
    minPct: number;
    meanPct: number;
    maxPct: number;
  };
}

export interface HighResSpecimen {
  id: string;
  title: string;
  defect_class: DefectClass;
  is_defective: boolean;
  resolution: string;
  substrate: string;
  severity: 'NONE' | 'MODERATE' | 'MAJOR' | 'CRITICAL';
  confidence: number;
  anomaly_score: number;
  dimensions_mm: {
    length: number;
    width: number;
  };
  bbox: [number, number, number, number];
  defect_pixels: number;
  description: string;
  files: {
    raw: string;
    mask: string;
    annotated: string;
  };
}

export interface HighResCatalog {
  generator_version: string;
  resolution: { width: number; height: number };
  total_specimens: number;
  specimens: HighResSpecimen[];
}
