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
 * g++ img-search.cpp -o img-search -std=c++17 \
 * -Wall -pthread `pkg-config --cflags --libs opencv4` -O3
 * 
 * or:
 * 
 * clang++ img-search.cpp -o img-search \
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
#include <cstring>
#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/img_hash.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/**
 * Prints the help message
 */
void print_help(){
	std::cout << "img-search usage:\n\n";
	std::cout << "img-search [files...]\n";
	std::cout << "img-search -t [threshold] [files...]\n";
	std::cout << "img-search -h\n\n";
	std::cout << "The filenames for comparison are read from stdin.\n";
	
}

// Mutex for the calculate_hash_values function
std::mutex mu;

/**
 * Calculate the perceptual hash of the images
 * 
 * @param file_list List of filenames for all images
 * @param hash_list Stores the hash values
 * @param hash_func Hash function
 * @param thread_id Number of the particular thread
 * @param num_threads Total number of threads
 */
void calculate_hash_values( const std::deque<std::string>& file_list, 
	std::map<unsigned long, cv::Mat>& hash_list, 
	cv::Ptr<cv::img_hash::ImgHashBase> hash_func, 
	unsigned int thread_id, unsigned int num_threads ){
	
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
		
		hash_list.insert( std::pair<unsigned long, cv::Mat>( i, 
		current_hash ) );
		
		mu.unlock();
		
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
	
	// this is used later
	int skip_argv = 1;
	
	// print help
	if( argc == 1 || ( argc >= 1 && strcmp(argv[1], "-h") == 0 ) ){
		print_help();
		return 0;
	}
	
	// this is the threshold under which images are considered similar
	double threshold = 2.0;
	if( argc >= 3 && ( strcmp(argv[1], "-t") == 0 ) ){
		
		try{
			threshold = stod( argv[2] );
		} catch( exception &e ){
			cerr << "Exception caught: " << e.what() << "\n";
		}
		
		skip_argv = 3;
	}
	
	// get list of filenames to search for and calculate their hashes
	//******************************************************************
	deque<string> search_list;
	for( int i = skip_argv; i < argc; i ++ )
		search_list.push_back( argv[i] );
	
	map<unsigned long, cv::Mat> search_hash_values;
	calculate_hash_values( search_list, search_hash_values,
		cv::img_hash::PHash::create(), 0, 1 );
	
	// get list of filenames to search in
	//******************************************************************
	
	// Stores the filenames
	// To save memory, each file is identified by an unsigned long
	// instead of a string.
	deque<string> file_list;
	
	string filename;
	while( getline( cin, filename ) ){
		file_list.push_back( filename );
	}	
	
	// calculate perceptual hash for each file and store them in map
	//******************************************************************
	
	// Stores the perceptual hash for all images.
	map<unsigned long, cv::Mat> img_hash_values;
	
	// create threads
    unsigned int num_threads = (thread::hardware_concurrency()!=0) ?
		thread::hardware_concurrency() : 1 ;
		
    thread t[num_threads];
    for( unsigned int i = 0; i < num_threads; ++i ){
		t[i] = thread( calculate_hash_values, ref(file_list), 
		ref(img_hash_values), cv::img_hash::PHash::create(), 
		i, num_threads );
	}
    
    // join threads
    for( unsigned int i = 0; i < num_threads; ++i ){
		t[i].join();
	}
	
	// check for similar images
	//******************************************************************
	
	set< unsigned long > results;
	
	// hash function used for comparison of two hashes
	cv::Ptr<cv::img_hash::ImgHashBase> hash_func = 
		cv::img_hash::PHash::create();
	
	for( auto it1 = img_hash_values.begin(); it1 != 
		img_hash_values.end(); it1++ ){
		
		for( auto it2 = search_hash_values.begin(); it2 !=
			search_hash_values.end(); it2++ ){
			
			if( hash_func->compare( img_hash_values[it1->first], 
				search_hash_values[it2->first] ) <= threshold ){
				
				results.emplace( it1->first );
			}
			
		}
		
	}
	
	// print results
	//******************************************************************
	
	for( auto r : results )
		cout << file_list[r] << "\n";
	
	return 0;
}
