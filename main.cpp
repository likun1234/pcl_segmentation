#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/console/time.h>
using namespace std;
typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

void regiongrowing(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	pcl::PCDWriter writer;
	pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod (tree);
	normal_estimator.setInputCloud (cloud);
	normal_estimator.setKSearch (50);
	normal_estimator.compute (*normals);

	pcl::IndicesPtr indices (new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0, 1.0);
	pass.filter (*indices);

	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	reg.setMinClusterSize (50);
	reg.setMaxClusterSize (1000000);
	reg.setSearchMethod (tree);
	reg.setNumberOfNeighbours (30);
	reg.setInputCloud (cloud);
	//reg.setIndices (indices);
	reg.setInputNormals (normals);
	reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
	reg.setCurvatureThreshold (1.0);

	std::vector <pcl::PointIndices> clusters;
	reg.extract (clusters);

	std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
	std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;
	std::cout << "These are the indices of the points of the initial" <<
		std::endl << "cloud that belong to the first cluster:" << std::endl;
	int counter = 0;
	while (counter < clusters[0].indices.size ())
	{
		std::cout << clusters[0].indices[counter] << ", ";
		counter++;
		if (counter % 10 == 0)
			std::cout << std::endl;
	}
	std::cout << std::endl;

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
	//writer("result.pcd", *colored_cloud);
	pcl::io::savePCDFile("result.pcd", *colored_cloud); 
	pcl::visualization::CloudViewer viewer ("Cluster viewer");
	viewer.showCloud(colored_cloud);
	while (!viewer.wasStopped ())
	{
	}
}


void cluster_segment(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	// Read in the cloud data
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
	std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	vg.setInputCloud (cloud);
	vg.setLeafSize (0.01f, 0.01f, 0.01f);
	vg.filter (*cloud_filtered);
	std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::PCDWriter writer;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.02);

	int i=0, nr_points = (int) cloud_filtered->points.size ();
	while (cloud_filtered->points.size () > 0.3 * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud (cloud_filtered);
		seg.segment (*inliers, *coefficients);
		if (inliers->indices.size () == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);

		// Get the points associated with the planar surface
		extract.filter (*cloud_plane);
		std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_f);
		*cloud_filtered = *cloud_f;
	}

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (0.02); // 2cm
	ec.setMinClusterSize (100);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_filtered);
	ec.extract (cluster_indices);

	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
			cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".pcd";
		writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
		j++;
	}

}


void sacsegment(pcl::PointCloud<PointT>::Ptr cloud)
{
	pcl::PCDReader reader;
	pcl::PassThrough<PointT> pass;
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
	pcl::PCDWriter writer;
	pcl::ExtractIndices<PointT> extract;
	pcl::ExtractIndices<pcl::Normal> extract_normals;
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

	// Datasets
	//pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
	pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

	// Read in the cloud data
	//reader.read (argv[1], *cloud);
	std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;

	// Build a passthrough filter to remove spurious NaNs
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0, 1.5);
	pass.filter (*cloud_filtered);
	std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

	// Estimate point normals
	ne.setSearchMethod (tree);
	ne.setInputCloud (cloud_filtered);
	ne.setKSearch (50);
	ne.compute (*cloud_normals);

	// Create the segmentation object for the planar model and set all the parameters
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
	seg.setNormalDistanceWeight (0.1);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.03);
	seg.setInputCloud (cloud_filtered);
	seg.setInputNormals (cloud_normals);
	// Obtain the plane inliers and coefficients
	seg.segment (*inliers_plane, *coefficients_plane);
	std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

	// Extract the planar inliers from the input cloud
	extract.setInputCloud (cloud_filtered);
	extract.setIndices (inliers_plane);
	extract.setNegative (false);

	// Write the planar inliers to disk
	pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
	extract.filter (*cloud_plane);
	std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
	writer.write ("table_scene_mug_stereo_textured_plane.pcd", *cloud_plane, false);

	// Remove the planar inliers, extract the rest
	extract.setNegative (true);
	extract.filter (*cloud_filtered2);
	extract_normals.setNegative (true);
	extract_normals.setInputCloud (cloud_normals);
	extract_normals.setIndices (inliers_plane);
	extract_normals.filter (*cloud_normals2);

	// Create the segmentation object for cylinder segmentation and set all the parameters
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_CYLINDER);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setNormalDistanceWeight (0.1);
	seg.setMaxIterations (10000);
	seg.setDistanceThreshold (0.05);
	seg.setRadiusLimits (0, 0.1);
	seg.setInputCloud (cloud_filtered2);
	seg.setInputNormals (cloud_normals2);

	// Obtain the cylinder inliers and coefficients
	seg.segment (*inliers_cylinder, *coefficients_cylinder);
	std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

	// Write the cylinder inliers to disk
	extract.setInputCloud (cloud_filtered2);
	extract.setIndices (inliers_cylinder);
	extract.setNegative (false);
	pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
	extract.filter (*cloud_cylinder);
	if (cloud_cylinder->points.empty ()) 
		std::cerr << "Can't find the cylindrical component." << std::endl;
	else
	{
		std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
		writer.write ("table_scene_mug_stereo_textured_cylinder.pcd", *cloud_cylinder, false);
	}

}

void color_segment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	pcl::search::Search <pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);

	pcl::IndicesPtr indices (new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0, 1.0);
	pass.filter (*indices);

	pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
	reg.setInputCloud (cloud);
	reg.setIndices (indices);
	reg.setSearchMethod (tree);
	reg.setDistanceThreshold (10);
	reg.setPointColorThreshold (6);
	reg.setRegionColorThreshold (5);
	reg.setMinClusterSize (600);

	std::vector <pcl::PointIndices> clusters;
	reg.extract (clusters);

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
	pcl::PCDWriter writer;
	writer.write("colored_cloud.pcd", *colored_cloud);
	pcl::visualization::CloudViewer viewer ("Cluster viewer");
	viewer.showCloud (colored_cloud);
	while (!viewer.wasStopped ())
	{
		boost::this_thread::sleep (boost::posix_time::microseconds (100));
	}
}

void MinCutSegmentation(pcl::PointCloud<PointT>::Ptr cloud)
{
	pcl::IndicesPtr indices (new std::vector <int>);
	pcl::PassThrough<PointT> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0, 1.0);
	pass.filter (*indices);

	pcl::MinCutSegmentation<PointT> seg;
	seg.setInputCloud (cloud);
	seg.setIndices (indices);

	pcl::PointCloud<PointT>::Ptr foreground_points(new pcl::PointCloud<PointT> ());
	PointT point;
	point.x = 68.97;
	point.y = -18.55;
	point.z = 0.57;
	foreground_points->points.push_back(point);
	seg.setForegroundPoints (foreground_points);

	seg.setSigma (0.25);
	seg.setRadius (3.0433856);
	seg.setNumberOfNeighbours (14);
	seg.setSourceWeight (0.8);

	std::vector <pcl::PointIndices> clusters;
	seg.extract (clusters);

	std::cout << "Maximum flow is " << seg.getMaxFlow () << std::endl;

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = seg.getColoredCloud ();
	pcl::PCDWriter writer;
	writer.write("min_cut.pcd", *colored_cloud);
	pcl::visualization::CloudViewer viewer ("Cluster viewer");
	viewer.showCloud(colored_cloud);
	while (!viewer.wasStopped ())
	{
	}
}

typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

	bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
	if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
		return (true);
	else
		return (false);
}

	bool
enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
	Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
	if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
		return (true);
	if (fabs (point_a_normal.dot (point_b_normal)) < 0.05)
		return (true);
	return (false);
}

	bool
customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
	Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
	if (squared_distance < 10000)
	{
		if (fabs (point_a.intensity - point_b.intensity) < 8.0f)
			return (true);
		if (fabs (point_a_normal.dot (point_b_normal)) < 0.06)
			return (true);
	}
	else
	{
		if (fabs (point_a.intensity - point_b.intensity) < 3.0f)
			return (true);
	}
	return (false);
}


void conditional_euclidean(pcl::PointCloud<PointTypeIO>::Ptr cloud_in )
{
	// Data containers used
	pcl::PointCloud<PointTypeIO>::Ptr cloud_out (new pcl::PointCloud<PointTypeIO>);
	pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>);
	pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
	pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);
	pcl::console::TicToc tt;

	// Load the input point cloud
	std::cerr << "Loading...\n", tt.tic ();
	pcl::io::loadPCDFile ("Statues_4.pcd", *cloud_in);
	std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_in->points.size () << " points\n";

	// Downsample the cloud using a Voxel Grid class
	std::cerr << "Downsampling...\n", tt.tic ();
	pcl::VoxelGrid<PointTypeIO> vg;
	vg.setInputCloud (cloud_in);
	vg.setLeafSize (80.0, 80.0, 80.0);
	vg.setDownsampleAllData (true);
	vg.filter (*cloud_out);
	std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_out->points.size () << " points\n";

	// Set up a Normal Estimation class and merge data in cloud_with_normals
	std::cerr << "Computing normals...\n", tt.tic ();
	pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
	pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
	ne.setInputCloud (cloud_out);
	ne.setSearchMethod (search_tree);
	ne.setRadiusSearch (300.0);
	ne.compute (*cloud_with_normals);
	std::cerr << ">> Done: " << tt.toc () << " ms\n";

	// Set up a Conditional Euclidean Clustering class
	std::cerr << "Segmenting to clusters...\n", tt.tic ();
	pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
	cec.setInputCloud (cloud_with_normals);
	cec.setConditionFunction (&customRegionGrowing);
	cec.setClusterTolerance (500.0);
	cec.setMinClusterSize (cloud_with_normals->points.size () / 1000);
	cec.setMaxClusterSize (cloud_with_normals->points.size () / 5);
	cec.segment (*clusters);
	cec.getRemovedClusters (small_clusters, large_clusters);
	std::cerr << ">> Done: " << tt.toc () << " ms\n";

	// Using the intensity channel for lazy visualization of the output
	for (int i = 0; i < small_clusters->size (); ++i)
		for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
			cloud_out->points[(*small_clusters)[i].indices[j]].intensity = -2.0;
	for (int i = 0; i < large_clusters->size (); ++i)
		for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
			cloud_out->points[(*large_clusters)[i].indices[j]].intensity = +10.0;
	for (int i = 0; i < clusters->size (); ++i)
	{
		int label = rand () % 8;
		for (int j = 0; j < (*clusters)[i].indices.size (); ++j)
			cloud_out->points[(*clusters)[i].indices[j]].intensity = label;
	}

	// Save the output point cloud
	std::cerr << "Saving...\n", tt.tic ();
	pcl::io::savePCDFile ("output.pcd", *cloud_out);
	std::cerr << ">> Done: " << tt.toc () << " ms\n";
}



	int
main (int argc, char** argv)
{
	pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>());
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PCDReader reader;
	reader.read (argv[1], *cloud);
	//sacsegment(cloud);
	//cluster_segment(cloud);
	//regiongrowing(cloud);
	//color_segment(cloud);
	MinCutSegmentation(cloud);
	return 0;
}

