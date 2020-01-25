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
 * g++ img-similarity-cluster.cpp -o img-similarity-cluster -std=c++17 -Wall -pthread `pkg-config --cflags --libs opencv4` -O3
 * or:
 * clang++ img-similarity-cluster.cpp -o img-similarity-cluster -std=c++17 -Wall -pthread `pkg-config --cflags --libs opencv4` -O3
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

// help message
#define print_help() \
	printf("img-similarity-cluster usage:\n"); \
	printf("-h\tshow this message\n"); \
	printf("-d=arg\tdirectory of images\n"); \
	printf("-r\tload images recursively\n"); \
	printf("-t=arg\tthreshold for similarity\n"); \
	printf("-u\tshow unique images\n");

// thread function for calculating perceptual hashes
std::mutex mu;
void calculate_hash_values( const std::deque<std::string>& file_list, std::map<unsigned long, cv::Mat>& hash_list,
	cv::Ptr<cv::img_hash::ImgHashBase> hash_func, unsigned int thread_id, unsigned int num_threads ){
	
	// iterate over file_list
	for( unsigned long i = 0; i < file_list.size(); ++i ){
		
		// check if correct thread for image
		if( i%num_threads != thread_id )
			continue;
		
		// read image
		cv::Mat current_image = cv::imread( file_list.at(i) );
		
		// check for image data
		if( !current_image.data )
			continue;
		
		// calculate hash	
		cv::Mat current_hash;
		hash_func->compute( current_image, current_hash );
		
		// store result
		mu.lock();
		hash_list.insert( std::pair<unsigned long, cv::Mat>( i, current_hash ) );
		mu.unlock();
		
	}
	
}

// recursion function for building the temporary image cluster (depth-first search)
void build_temp_cluster( std::set< unsigned long >& temp_cluster,
	std::map< unsigned long, std::set< unsigned long > >& image_similarities, unsigned long start ){
	
	for( auto& i : image_similarities.at(start) ){
		
		// is element already in temp_cluster ?
		if( temp_cluster.find(i) != temp_cluster.end() )
			continue;
		
		// new element:
			temp_cluster.emplace(i);
			build_temp_cluster( temp_cluster, image_similarities, i );
	}
	
}

// main function
int main( int argc, char* argv[] ){
	
	using namespace std;
	namespace fs = filesystem;
	
	// check arguments
	int c;
	bool be_recursive = false, show_unique = false;
	bool flag_directory = false, flag_threshold = false;
	string string_threshold, string_directory;
	while( ( c = getopt( argc, argv, "hrud:t:") ) != -1 ){
		
		switch(c){
			case 'h':
				print_help();
				return 0;
				break;
			case 'r':
				be_recursive = true;
				break;
			case 'u':
				show_unique = true;
				break;
			case 'd':
				flag_directory = 1;
				string_directory = optarg;
				break;
			case 't':
				flag_threshold = 1;
				string_threshold = optarg;
				break;
			default:
				break;
		}
		
	}
	
	if( !flag_directory ){
		cout << "Error: missing argument -d\n";
		return 0;
	}
	
	double threshold = 2.0;
	if( flag_threshold ){
		try{
			threshold = stod( string_threshold );
		} catch( exception &e ){
			
		}
	}
	
	// get list of filenames 
	// to save memory, each file is identified by an unsigned long instead of a string
	deque<string> file_list;
    fs::path directory_path = string_directory;
    
    // check if path is directory
    if( !( fs::exists( directory_path ) && fs::is_directory( directory_path ) ) ){
		cout << "Error: Couldn't open " << directory_path << endl;
		return 0;
	}
    
    // load filenames
    if( be_recursive ){
		// recurse directory and add filenames to deque
		for( auto p: fs::recursive_directory_iterator( directory_path ) ){
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
	
	cout << "Filelist created, " << file_list.size() << " files.\n";
	
	
	
	// calculate perceptual hash for each file and store them in map
	// file -> hash
	map<unsigned long, cv::Mat> img_hash_values;
	
	// create threads
    unsigned int num_threads = (thread::hardware_concurrency()!=0) ? thread::hardware_concurrency() : 1 ;
    thread t[num_threads];
    for( unsigned int i = 0; i < num_threads; ++i ){
		t[i] = thread( calculate_hash_values, ref(file_list), ref(img_hash_values), cv::img_hash::PHash::create(), i, num_threads );
	}
    
    // join threads
    for( unsigned int i = 0; i < num_threads; ++i ){
		t[i].join();
	}
	
	cout << "Finished hash calculation.\n";
	
	
	
	// create a map of all unique file pairs to their hash difference
	map< pair< unsigned long, unsigned long >, double > image_deltas;
	// hash function used for comparison of two hashes
	cv::Ptr<cv::img_hash::ImgHashBase> hash_func = cv::img_hash::PHash::create();
	
	for( auto it1 = img_hash_values.begin(); it1 != img_hash_values.end(); it1++ ){
		
		auto it2 = it1;
		it2++;
		for( ; it2 != img_hash_values.end(); it2++ ){
			image_deltas[std::pair< unsigned long, unsigned long >( it1->first, it2->first )] =
				hash_func->compare( img_hash_values[it1->first], img_hash_values[it2->first] );
		}
		
	}
	
	cout << "Hash distances calculated, " << image_deltas.size() << " image pairs.\n";
	
	
	
	// create map of images to their similar images (adjacent vertices in the graph)
	map< unsigned long, set< unsigned long > > image_similarities;
	
	// fill map with images
	for( auto& i : img_hash_values ){
		image_similarities[i.first] = set< unsigned long >();
	}
	
	// check for similar image pairs
	for( auto& i : image_deltas ){
		// images are similar
		if( i.second <= threshold ){
			image_similarities[i.first.first].emplace(i.first.second);
			image_similarities[i.first.second].emplace(i.first.first);
		}
	}
	
	cout << "Adjacency lists created.\n";
	
	
	
	// get image clusters (graph components) and unique images
	vector< set< unsigned long > > image_clusters;
	set<unsigned long> unique_images;
	
	for( auto& i : image_similarities ){
		
		if( i.second.empty() ){ // no similarities
			unique_images.emplace( i.first );
			continue;
		}
		
		// recursively create set of connected images 
		set< unsigned long > temp_cluster;
		temp_cluster.emplace( i.first );
		build_temp_cluster( temp_cluster, image_similarities, i.first );
		bool merged = false;
		
		// unify temp_cluster with existing clusters
		for( auto& j : image_clusters ){
			
			// if common elements merge temp_cluster with image cluster
			if( find_first_of( j.begin(), j.end(), temp_cluster.begin(), temp_cluster.end() ) != j.end() ){
				for( auto& k : temp_cluster ){
					j.emplace( k );
				}
				merged = true;
				break;
			}
			
		}
		
		// is temp_cluster merged with image_clusters, no:
		if( !merged ){
			image_clusters.push_back( temp_cluster );
		}
		//cout << "checked " << i.first << "\n";
	}
	
	// print image clusters
	for( unsigned int i = 0; i < image_clusters.size(); i++ ){
		cout << "image cluster " << i << ":\n";
		for( auto& j : image_clusters[i] ){
			cout << file_list[j] << "\n";
		}
	}
	
	// print unique images
	if( show_unique ){
		cout << "unique images:\n";
		for( auto& i : unique_images ){
			cout << file_list[i] << "\n";
		}
	}
}
