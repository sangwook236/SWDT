#include <scip/scip.h>
#include <scip/scipshell.h>
#include <scip/scipdefplugins.h>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// REF [file] >> ${SCIP_HOME}/examples/Binpacking/src/cmain.c
SCIP_RETCODE runShell(
	int argc,  // Number of shell parameters.
	char** argv,  // Array with shell parameters.
	const char* defaultsetname  // Name of default settings file.
)
{
#if 0
	SCIP* scip = NULL;

	// Initialize SCIP.
	SCIP_CALL(SCIPcreate(&scip));

	// we explicitly enable the use of a debug solution for this main SCIP instance.
	SCIPenableDebugSol(scip);

	// Include binpacking reader.
	SCIP_CALL(SCIPincludeReaderBpa(scip));

	// Include binpacking branching and branching data.
	SCIP_CALL(SCIPincludeBranchruleRyanFoster(scip));
	SCIP_CALL(SCIPincludeConshdlrSamediff(scip));

	// Include binpacking pricer.
	SCIP_CALL(SCIPincludePricerBinpacking(scip));

	// Include default SCIP plugins.
	SCIP_CALL(SCIPincludeDefaultPlugins(scip));

	// For column generation instances, disable restarts.
	SCIP_CALL(SCIPsetIntParam(scip, "presolving/maxrestarts", 0));

	// Turn off all separation algorithms.
	SCIP_CALL(SCIPsetSeparating(scip, SCIP_PARAMSETTING_OFF, TRUE));

	// Process command line arguments.
	SCIP_CALL(SCIPprocessShellArguments(scip, argc, argv, defaultsetname));

	// DeinitializE.
	SCIP_CALL(SCIPfree(&scip));

	BMScheckEmptyMemory();

	return SCIP_OKAY;
#else
	throw std::runtime_error("Not yet implemented");
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_scip {

// REF [file] >> ${SCIP_HOME}/examples/Binpacking/src/cmain.c
void bin_packing_example(int argc, char *argv[])
{
	const SCIP_RETCODE retcode = local::runShell(argc, argv, "scip.set");
	if (SCIP_OKAY != retcode)
	{
		//SCIPprintError(retcode);  // NOTICE [error] >> Unresolved external symbol _SCIPprintError.
	}
}


}  // namespace my_scip
