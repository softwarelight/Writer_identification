#pragma once

#include <string>
#include <vector>


class FsHelpers
{
public:
    // Path Composition

    // combine path, e.g. "/foo", "bar" --> "/foo/bar"
    static std::string CombinePath(
        const std::string& path1,
        const std::string& path2);

    static std::string CombinePath(
        const std::string& path1,
        const std::string& path2,
        const std::string& path3);

    static std::string CombinePath(
        const std::string& path1,
        const std::string& path2,
        const std::string& path3,
        const std::string& path4);

    static std::string CombinePath(const std::vector<std::string>& paths);


    // Path Decomposition

    // get file name, e.g. "/foo/bar.xml" --> "bar.xml"
    static std::string GetFileName(const std::string& path);

    // get file extension with dot, e.g. "/foo/bar.xml" --> ".xml"
    static std::string GetExtension(const std::string& path);

    // get file name witout extension, e.g. "/foo/bar.xml" --> "bar"
    static std::string GetFileNameWithoutExtension(const std::string& path);

    // get parent path, e.g. "/my/foo/bar.xml" --> "/my/foo"
    static std::string GetParentPath(const std::string& path);

    // get containing directory name, e.g. "/my/foo/bar.xml" --> "foo"
    static std::string GetParentName(const std::string& path);


    // Path Operation

    // return true if path(directory or regular file) exists
    static bool Exists(const std::string& path);

    // create directory, return false if parent path not exist
    static bool MakeDirectory(const std::string& path);

    // create directory and its parent direcotries that does not exist
    static bool MakeDirectories(const std::string& path);

    // get all directories under directory non-recursively
    static void GetDirectories(
        const std::string& path,
        std::vector<std::string>& out_paths);

    // get all files that match regex under directory (recursively)
    static void GetFilesMatchRegex(
        const std::string& path,
        std::vector<std::string>& out_paths,
        const std::string& filterRegex = "*",
        bool recursively = false);

    // get all files that has extension under directory (recursively)
    static void GetFilesHasExtension(
        const std::string& path,
        std::vector<std::string>& out_paths,
        const std::string& extension,
        bool recursively = false);


    // File Operation

    // read all lines of a text file
    static bool ReadAllLines(const std::string& filePath, std::vector<std::string>& out_allLines);

private:
    // pure static class, no constructor
    FsHelpers();
};
