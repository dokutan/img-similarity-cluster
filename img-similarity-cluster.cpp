/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 */

/* compile with:
 * 
 * g++ img-similarity-cluster.cpp -o img-similarity-cluster -std=c++17 \
 * -Wall -pthread `pkg-config --cflags --libs opencv4` -O3
 * 
 * or:
 * 
 * clang++ img-similarity-cluster.cpp -o img-similarity-cluster \
 * -std=c++17 -Wall -pthread `pkg-config --cflags --libs opencv4` -O3
 */

#include <iostream>
#include <fstream>
#include <deque>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <algorithm>
#include <exception>
#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/imgcodecs.hpp>
#include "opencv2/img_hash.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/**
 * Prints the help message
 */
#define print_help() \
	printf("Finds groups of similar images.\n\n"); \
	printf("img-similarity-cluster usage:\n\n"); \
	printf("-h\tshow this message\n"); \
	printf("-d=arg\tdirectory of images (- for stdin)\n"); \
	printf("-r\tload images recursively\n"); \
	printf("-t=arg\tthreshold for similarity\n"); \
	printf("-l\tprint all similar images on one line and nothing else\n");


// Mutex for the calculate_hash_values function
std::mutex mu;

/**
 * Calculate the perceptual hash of the images
 * 
 * @param file_list List of filenames for all images
 * @param hash_list Stores the hash values
 * @param thread_id Number of the particular thread
 * @param num_threads Total number of threads
 */
void calculate_hash_values( const std::deque<std::string>& file_list, 
	std::vector< cv::Mat >& hash_list,
	unsigned int thread_id, unsigned int num_threads ){

	cv::Ptr<cv::img_hash::ImgHashBase> hash_func = cv::img_hash::PHash::create();
	
	// iterate over file_list
	for( unsigned long i = 0; i < file_list.size(); i++ ){
		
		// check if correct thread for image
		if( i%num_threads != thread_id )
			continue;
		
		cv::Mat current_image, current_hash;
		
		// read image
		current_image = cv::imread( file_list.at(i) );
		
		// check for image data
		if( !current_image.data ){
			hash_list.at(i) = current_image;
			continue;
		}
		
		// calculate hash
		hash_func->compute( current_image, current_hash );

		// store hash
		hash_list.at(i) = current_hash;
	}
}

/**
 * Calculate all similar pairs of images
 * 
 * @param hash_list List of all hash values
 * @param similar_pairs Stores the similar pairs
 * @param threshold Similarity threshold
 * @param thread_id Number of the particular thread
 * @param num_threads Total number of threads
 */
void calculate_similar_pairs(const std::vector< cv::Mat >& hash_list,
	std::map< unsigned long, std::set< unsigned long > >& image_similarities,
	double threshold,
	unsigned int thread_id, unsigned int num_threads ){
	
	// hash function used for comparison of two hashes
	cv::Ptr<cv::img_hash::ImgHashBase> hash_func = cv::img_hash::PHash::create();

	// iterate over hash_list
	for( unsigned long i = 0; i < hash_list.size(); i++ ){
		
		// check if correct thread for image
		if( i%num_threads != thread_id )
			continue;
		
		if( !hash_list.at(i).data )
			continue;
		
		for( unsigned long j = i+1; j < hash_list.size(); ++j ){
			if( !hash_list.at(j).data )
				continue;

			if(hash_func->compare( hash_list.at(i), hash_list.at(j) ) <= threshold ){
				mu.lock();
				if(!image_similarities.contains(i)){
					image_similarities.emplace(i, std::set<unsigned long>());
				}
				image_similarities.at(i).emplace(j);
				mu.unlock();
			}
		}
		
	}
}

/**
 * Recursion function for building the temporary image cluster
 * (depth-first search)
 */
void build_temp_cluster( std::set< unsigned long >& temp_cluster,
	std::map< unsigned long, std::set< unsigned long > >
	& image_similarities, unsigned long start ){
	
	if(!image_similarities.contains(start)){
		return;
	}

	for( auto& i : image_similarities.at(start) ){
		
		// is element already in temp_cluster ?
		if( temp_cluster.contains(i) )
			continue;
		
		// new element:
		temp_cluster.emplace(i);
		build_temp_cluster( temp_cluster, image_similarities, i );
	}
	
}

/**
 * Main function
 */
int main( int argc, char* argv[] ){
	
	using namespace std;
	namespace fs = filesystem;
	
	
	
	// check arguments
	//******************************************************************
	
	int c;
	bool be_recursive = false, one_line = false;
	bool flag_directory = false, flag_threshold = false;
	string string_threshold, string_directory;
	while( ( c = getopt( argc, argv, "hrd:t:l") ) != -1 ){
		
		switch(c){
			case 'h':
				print_help();
				return 0;
				break;
			case 'r':
				be_recursive = true;
				break;
			case 'd':
				flag_directory = 1;
				string_directory = optarg;
				break;
			case 't':
				flag_threshold = 1;
				string_threshold = optarg;
				break;
			case 'l':
				one_line = true;
				break;
			default:
				break;
		}
		
	}
	
	// check if the directory is specified on the commandline
	if( !flag_directory ){
		cout << "Error: missing argument -d\n";
		return 0;
	}
	
	// this is the threshold, under which images are considered similar
	double threshold = 0.2;
	
	// check if the threshold is explicitly specified on the commandline
	if( flag_threshold ){
		try{
			threshold = stod( string_threshold );
		} catch( exception &e ){
			
		}
	}

	// create threads
    //******************************************************************
	unsigned int num_threads = (thread::hardware_concurrency()!=0) ?
		thread::hardware_concurrency() : 1 ;
		
    std::vector< thread > t;
	t.resize(num_threads);	
	
	
	// get list of filenames
	//******************************************************************
	
	// Stores the filenames
	// To save memory, each file is identified by an unsigned long
	// instead of a string.
	deque<string> file_list;
    fs::path directory_path = string_directory;
    
    if( directory_path == "-" ){ // load filenames from stdin
		
		string filename;
		while( getline( cin, filename ) ){
			file_list.push_back( filename );
		}
		
	} else{ // load from directory
		// check if path is directory
		if( !( fs::exists( directory_path ) &&
			fs::is_directory( directory_path ) ) ){
			
			cout << "Error: Couldn't open " << directory_path << endl;
			return 0;
		}
		
		// load filenames
		if( be_recursive ){
			
			// recurse directory and add filenames to deque
			for( auto p:
				fs::recursive_directory_iterator( directory_path ) ){
				
				if( fs::is_regular_file(p.path()) ) {
					file_list.push_back( p.path().string() );
				}
				
			}
			
		} else{
			
			for( auto p: fs::directory_iterator( directory_path ) ){
				
				if( fs::is_regular_file(p.path()) ) {
					file_list.push_back( p.path().string() );
				}
				
			}
			
		}
	}
	
	if(!one_line)
		cout << "Filelist created, " << file_list.size() << " files.\n";
	
	
	
	// calculate perceptual hash for each file
	//******************************************************************
	
	std::vector< cv::Mat > hash_list;

	hash_list.resize( file_list.size() );
    for( unsigned int i = 0; i < num_threads; ++i ){
		t.at(i) = thread( calculate_hash_values, ref(file_list), ref(hash_list), i, num_threads );
	}
    for( unsigned int i = 0; i < num_threads; ++i ){
		t.at(i).join();
	}

	if(!one_line)
		cout << "Finished hash calculations.\n";
	
	
	// create map of images to their similar images
	//******************************************************************

	map< unsigned long, set< unsigned long > > image_similarities;

	for( unsigned int i = 0; i < num_threads; ++i ){
		t.at(i) = thread( calculate_similar_pairs, ref(hash_list), ref(image_similarities), threshold, i, num_threads );
	}
    for( unsigned int i = 0; i < num_threads; ++i ){
		t.at(i).join();
	}

	// hashes are no longer needed
	hash_list.clear();
	hash_list.shrink_to_fit();

	if(!one_line)
		cout << "Adjacency lists created.\n";
	
	
	// get image clusters (graph components) and unique images
	//******************************************************************
	
	vector< set< unsigned long > > image_clusters;
	
	for( auto& i : image_similarities ){
		
		if( i.second.empty() ){ // no similarities
			std::cout << "ok1\n";
			continue;
		}
		
		// recursively create set of connected images 
		set< unsigned long > temp_cluster;
		
		temp_cluster.emplace( i.first );
		build_temp_cluster( temp_cluster, image_similarities, i.first );
		
		bool merged = false;
		
		// unify temp_cluster with existing clusters
		for( auto& j : image_clusters ){
			
			// if common elements: merge temp_cluster with image cluster
			if( find_first_of( j.begin(), j.end(), temp_cluster.begin(),
			temp_cluster.end() ) != j.end() ){
				
				for( auto& k : temp_cluster ){
					j.emplace( k );
				}
				merged = true;
				break;
				
			}
			
		}
		
		// is temp_cluster merged with image_clusters, if not,
		// add temp_cluster as new cluster to image_clusters
		if( !merged ){
			image_clusters.push_back( temp_cluster );
		}
		//cout << "checked " << i.first << "\n";
	}
	
	// print image clusters
	for( unsigned int i = 0; i < image_clusters.size(); i++ ){
		
		if(!one_line)
			cout << "image cluster " << i << ":\n";

		for( auto& j : image_clusters[i] ){
			cout << file_list.at(j) << (one_line ? "\t" : "\n");
		}

		if(one_line)
			cout << "\n";
		
	}

	return 0;
}
